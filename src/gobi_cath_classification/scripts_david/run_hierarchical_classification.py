from gobi_cath_classification.scripts_david.hierarchical_classifier import HierarchicalClassifier
from pathlib import Path

def main():
    hc = HierarchicalClassifier(models=[
        "C:\\Users\\David Mauder\\ray_results\\hc svm C\\svm_acc_c_0.92\\model_object.model",
        "C:\\Users\\David Mauder\\ray_results\\hc fcnn A\\fcnn_acc_a_0.735\\model_object.model",
        "C:\\Users\\David Mauder\\ray_results\\hc gnb T\\gnb_acc_t_0.33\\model_object.model",
        "C:\\Users\\David Mauder\\ray_results\\hc svm C\\svm_acc_c_0.92\\model_object.model"
    ], classifier_type="LCL", classification_cutoff=0)
    hc.get_data(random_seed=1)
    hc.predict_lcl()


if __name__ == "__main__":
    main()