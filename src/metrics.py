from torchmetrics import JaccardIndex, F1Score, MetricCollection, MetricTracker
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassAccuracy


def get_metrics_classification_multi(num_classes):

    list_of_metrics = {
        "JaccardIndex_macro": JaccardIndex(task="multiclass",num_classes=num_classes,average="macro"),
        "JaccardIndex_micro": JaccardIndex(task="multiclass",num_classes=num_classes,average="micro"),
        "F1Score_macro": F1Score(task="multiclass",num_classes=num_classes,average="macro"),
        "F1Score_micro": F1Score(task="multiclass",num_classes=num_classes,average="micro"),
        "Accuracy_macro": MulticlassAccuracy(num_classes=num_classes,average="macro"),
        "Accuracy_micro": MulticlassAccuracy(num_classes=num_classes,average="micro"),
        "Recall_macro": MulticlassRecall(num_classes=num_classes,average="macro"),
        "Recall_micro": MulticlassRecall(num_classes=num_classes,average="micro"),
        "Precision_macro": MulticlassPrecision(num_classes=num_classes,average="macro"),
        "Precision_micro": MulticlassPrecision(num_classes=num_classes,average="micro"),
    }  

    maximize_list=[True for _ in range(len(list_of_metrics))]

    metric_coll = MetricCollection(list_of_metrics)

    tracker = MetricTracker(metric_coll, maximize=maximize_list)

    return tracker


if __name__ == "__main__":
    tracker = get_metrics_classification_multi(3)