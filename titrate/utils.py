def calc_ts_toyMC(dataset, test_statistic, poi_val, poi_true_val, poi_name):
    """Evaluates given test statistic for toy MCs.

    The given dataset will be copied and a new measurement will be simulated.
    The test statistic is evaluated for this dataset.
    """
    toy_dataset = copy_dataset_with_models(dataset)
    toy_dataset.models.parameters[poi_name].value = poi_true_val
    toy_dataset.fake()

    ts = test_statistic(toy_dataset, poi_name)
    return ts.evaluate(poi_val)


def copy_dataset_with_models(dataset):
    """Copies a dataset inlcuding the models."""
    dataset_copy = dataset.copy()

    model_copies = dataset.models.copy()
    model_copies[0]._name = f"{dataset_copy.name}-signal"
    model_copies[1].datasets_names = [f"{dataset_copy.name}"]
    dataset_copy.models = model_copies
    return dataset_copy
