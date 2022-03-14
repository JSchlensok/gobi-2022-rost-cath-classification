from gobi_cath_classification.scripts_david.hierarchical_classifier import HierarchicalClassifier
from pathlib import Path

def main():
    hc = HierarchicalClassifier(models=[
        "C:\\Users\\David Mauder\\ray_results\\hc lcpn C\\C\\model_object.model",
        "C:\\Users\\David Mauder\\ray_results\\hc lcpn A\\A\\model_object.model",
        "C:\\Users\\David Mauder\\ray_results\\hc lcpn T\\T\\model_object.model",
        "C:\\Users\\David Mauder\\ray_results\\hc lcpn H\\H\\model_object.model"
    ], classifier_type="LCL", classification_cutoff=0)
    hc.get_data(random_seed=1)
    hc.predict_lcpn(threshold=0)


if __name__ == "__main__":
    main()