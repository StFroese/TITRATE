from gammapy.datasets import Datasets


class CopyModelError(AttributeError):
    """Something went wrong during copying a model."""
    pass


def calc_ts_toyMC(dataset, test_statistic, poi_val, poi_true_val, poi_name):
    """Evaluates given test statistic for toy MCs.

    The `poi_true_val` is used to set the parameter of interest in the model
    before simulating the new measurement from which the test statistic is
    evaluated.
    """
    toy_dataset = copy_dataset_with_models(dataset)
    toy_dataset.models.parameters[poi_name].value = poi_true_val
    toy_dataset.fake()

    ts = test_statistic(toy_dataset, poi_name)
    return ts.evaluate(poi_val)


def copy_dataset_with_models(dataset):
    """Copies a dataset inlcuding the models."""
    dataset_copy = dataset.copy()
    copy_models_to_dataset(dataset.models, dataset_copy)

    return dataset_copy


def copy_models_to_dataset(models, dataset):
    """Copies models and assigns them to dataset."""
    model_copies = models.copy()
    if isinstance(dataset, Datasets):
        for d in dataset:
            b = model_copies[1].copy()
            b.datasets_names = [d.name]
            d.models = [model_copies[0], b]
    else:
        for model in model_copies:
            if hasattr(model, "_name"):
                model._name = f"{dataset.name}-{model._name}"
            if hasattr(model, "datasets_names"):
                model.datasets_names = [dataset.name]
            else:
                raise CopyModelError(
                    f"{model.__class__.__name__} doesn't provided `._name`"
                    f"nor `.datasets_names`."
                )
        dataset.models = model_copies
