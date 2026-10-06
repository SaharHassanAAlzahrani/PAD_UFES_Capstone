from sklearn.model_selection import StratifiedGroupKFold


import torch
import torch.nn as nn

from pad_multimodal.config import (
    CLASS_TO_IDX,
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE,
    MAX_EPOCHS,
    N_INNER_FOLDS,
    NUM_CLASSES,
    SEED,
    WEIGHT_DECAY,
)

from pad_multimodal.evaluate import calculate_metrics
from pad_multimodal.train import (
    calculate_class_weights,
    predict,
    train_one_epoch,
)
from pad_multimodal.data import build_multimodal_loaders
from pad_multimodal.models.factory import build_model
from pad_multimodal.preprocessing import fit_transform_metadata
from pad_multimodal.reproducibility import set_fold_seed
from sentence_transformers import SentenceTransformer

from pad_multimodal.sentence_metadata import encode_metadata_sentences

def create_inner_split(
    outer_train_df,
    n_splits=N_INNER_FOLDS,
    seed=SEED,
):
    """Create one patient-grouped inner train/validation split."""

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    train_idx, val_idx = next(
        splitter.split(
            X=outer_train_df,
            y=outer_train_df["diagnostic"],
            groups=outer_train_df["patient_id"],
        )
    )

    inner_train_df = (
        outer_train_df
        .iloc[train_idx]
        .copy()
        .reset_index(drop=True)
    )

    inner_val_df = (
        outer_train_df
        .iloc[val_idx]
        .copy()
        .reset_index(drop=True)
    )

    train_patients = set(inner_train_df["patient_id"])
    val_patients = set(inner_val_df["patient_id"])

    if train_patients & val_patients:
        raise RuntimeError(
            "Patient overlap detected between inner train and validation."
        )

    return inner_train_df, inner_val_df

def create_outer_split(
    dataframe,
    outer_fold,
):
    """Separate one fixed outer fold for held-out evaluation."""

    outer_train_df = (
        dataframe[dataframe["fold"] != outer_fold]
        .copy()
        .reset_index(drop=True)
    )

    outer_test_df = (
        dataframe[dataframe["fold"] == outer_fold]
        .copy()
        .reset_index(drop=True)
    )

    if outer_train_df.empty:
        raise ValueError("Outer training partition is empty.")

    if outer_test_df.empty:
        raise ValueError("Outer held-out partition is empty.")

    train_patients = set(outer_train_df["patient_id"])
    test_patients = set(outer_test_df["patient_id"])

    if train_patients & test_patients:
        raise RuntimeError(
            "Patient overlap detected between outer train and held-out data."
        )

    return outer_train_df, outer_test_df

def select_training_epochs(
    model,
    train_loader,
    val_loader,
    train_df,
    device,
    max_epochs=MAX_EPOCHS,
    patience=EARLY_STOPPING_PATIENCE,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
):
    """Select the training epoch count using inner validation macro-F1."""

    train_targets = (
        train_df["diagnostic"]
        .map(CLASS_TO_IDX)
        .to_numpy()
    )

    class_weights = calculate_class_weights(
        targets=train_targets,
        num_classes=NUM_CLASSES,
        device=device,
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    best_macro_f1 = float("-inf")
    best_epoch = 1
    epochs_without_improvement = 0

    history = []

    model = model.to(device)

    for epoch in range(1, max_epochs + 1):
        learning_rate_used = optimizer.param_groups[0]["lr"]

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_results = predict(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        val_metrics = calculate_metrics(
            y_true=val_results["targets"],
            y_pred=val_results["predictions"],
            y_prob=val_results["probabilities"],
        )

        val_macro_f1 = val_metrics["macro_f1"]

        history.append({
            "epoch": epoch,
            "learning_rate": learning_rate_used,
            "train_loss": train_loss,
            "val_loss": val_results["loss"],
            "val_macro_f1": val_macro_f1,
            "val_balanced_accuracy": val_metrics[
                "balanced_accuracy"
            ],
        })

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        scheduler.step(val_macro_f1)

        if epochs_without_improvement >= patience:
            break

    learning_rate_schedule = [
        row["learning_rate"]
        for row in history[:best_epoch]
    ]

    return {
        "selected_epochs": best_epoch,
        "best_macro_f1": best_macro_f1,
        "learning_rate_schedule": learning_rate_schedule,
        "history": history,
    }

def train_final_model(
    model,
    train_loader,
    train_df,
    device,
    selected_epochs,
    learning_rate_schedule,
    weight_decay=WEIGHT_DECAY,
):
    """Train a fresh model on the full outer-training partition."""

    if selected_epochs < 1:
        raise ValueError(
            "selected_epochs must be at least 1."
        )

    if len(learning_rate_schedule) != selected_epochs:
        raise ValueError(
            "Learning-rate schedule length must match selected_epochs."
        )

    train_targets = (
        train_df["diagnostic"]
        .map(CLASS_TO_IDX)
        .to_numpy()
    )

    class_weights = calculate_class_weights(
        targets=train_targets,
        num_classes=NUM_CLASSES,
        device=device,
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate_schedule[0],
        weight_decay=weight_decay,
    )

    model = model.to(device)

    history = []

    for epoch in range(1, selected_epochs + 1):
        learning_rate = learning_rate_schedule[
            epoch - 1
        ]

        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        history.append({
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": train_loss,
        })

    return {
        "model": model,
        "criterion": criterion,
        "history": history,
    }

def evaluate_final_model(
    model,
    test_loader,
    criterion,
    device,
):
    """Evaluate the final model once on the outer held-out partition."""

    results = predict(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )

    metrics = calculate_metrics(
        y_true=results["targets"],
        y_pred=results["predictions"],
        y_prob=results["probabilities"],
    )

    metrics["loss"] = results["loss"]

    return {
        "metrics": metrics,
        "predictions": results,
    }

def prepare_metadata(
    train_df,
    eval_df,
    model_name,
):
    """Prepare the metadata representation required by a model."""

    if model_name == "metablock_se":
        sentence_encoder = SentenceTransformer(
            "sentence-transformers/paraphrase-albert-small-v2"
        )

        train_metadata = encode_metadata_sentences(
            train_df,
            sentence_encoder,
        )
        eval_metadata = encode_metadata_sentences(
            eval_df,
            sentence_encoder,
        )

        return (
            train_metadata,
            eval_metadata,
            sentence_encoder,
        )

    return fit_transform_metadata(
        train_df=train_df,
        eval_df=eval_df,
    )
    
def run_outer_fold(
    dataframe,
    image_lookup,
    model_name,
    outer_fold,
    device,
):
    """Run one complete nested outer-fold experiment."""

    fold_seed = set_fold_seed(
        SEED,
        outer_fold,
    )

    outer_train_df, outer_test_df = create_outer_split(
        dataframe=dataframe,
        outer_fold=outer_fold,
    )

    inner_train_df, inner_val_df = create_inner_split(
        outer_train_df=outer_train_df,
        seed=SEED,
    )

    (
        inner_train_metadata,
        inner_val_metadata,
        inner_preprocessor,
    ) = prepare_metadata(
        train_df=inner_train_df,
        eval_df=inner_val_df,
        model_name=model_name,
    )

    metadata_dim = inner_train_metadata.shape[1]

    inner_train_loader, inner_val_loader = (
        build_multimodal_loaders(
            train_df=inner_train_df,
            eval_df=inner_val_df,
            train_metadata=inner_train_metadata,
            eval_metadata=inner_val_metadata,
            image_lookup=image_lookup,
        )
    )

    development_model = build_model(
        model_name=model_name,
        metadata_dim=metadata_dim,
    )

    selection = select_training_epochs(
        model=development_model,
        train_loader=inner_train_loader,
        val_loader=inner_val_loader,
        train_df=inner_train_df,
        device=device,
    )

    del development_model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    (
        outer_train_metadata,
        outer_test_metadata,
        outer_preprocessor,
    ) = prepare_metadata(
        train_df=outer_train_df,
        eval_df=outer_test_df,
        model_name=model_name,
    )

    outer_metadata_dim = outer_train_metadata.shape[1]

    if outer_metadata_dim != metadata_dim:
        raise RuntimeError(
            "Metadata dimensionality changed between inner and outer preprocessing."
        )

    outer_train_loader, outer_test_loader = (
        build_multimodal_loaders(
            train_df=outer_train_df,
            eval_df=outer_test_df,
            train_metadata=outer_train_metadata,
            eval_metadata=outer_test_metadata,
            image_lookup=image_lookup,
        )
    )

    set_fold_seed(
        SEED,
        outer_fold,
    )

    final_model = build_model(
        model_name=model_name,
        metadata_dim=outer_metadata_dim,
    )

    final_training = train_final_model(
        model=final_model,
        train_loader=outer_train_loader,
        train_df=outer_train_df,
        device=device,
        selected_epochs=selection["selected_epochs"],
        learning_rate_schedule=selection[
            "learning_rate_schedule"
        ],
    )

    evaluation = evaluate_final_model(
        model=final_training["model"],
        test_loader=outer_test_loader,
        criterion=final_training["criterion"],
        device=device,
    )

    return {
        "model_name": model_name,
        "outer_fold": outer_fold,
        "fold_seed": fold_seed,
        "selected_epochs": selection[
            "selected_epochs"
        ],
        "inner_best_macro_f1": selection[
            "best_macro_f1"
        ],
        "inner_history": selection["history"],
        "final_history": final_training["history"],
        "metrics": evaluation["metrics"],
        "predictions": evaluation["predictions"],
        "metadata_dim": outer_metadata_dim,
        "n_outer_train": len(outer_train_df),
        "n_outer_test": len(outer_test_df),
    }