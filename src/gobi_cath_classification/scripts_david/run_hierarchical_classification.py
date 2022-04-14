from gobi_cath_classification.scripts_david.hierarchical_classifier import HierarchicalClassifier


def main():
    hc_lcpn = HierarchicalClassifier(
        models=[
            "C:\\Users\\David Mauder\\ray_results\\hc lcpn C\\C\\model_object.model",
            "C:\\Users\\David Mauder\\ray_results\\hc lcpn A\\A\\model_object.model",
            "C:\\Users\\David Mauder\\ray_results\\hc lcpn T\\T\\model_object.model",
            "C:\\Users\\David Mauder\\ray_results\\hc lcpn H\\H\\model_object.model",
        ],
        classifier_type="LCPN",
        classification_cutoff=0,
    )
    hc_lcpn.get_data(random_seed=42)
    hc_lcpn.predict_lcpn(threshold=0, prediction="AVG")

    hc_lcl = HierarchicalClassifier(
        models=[
            "C:\\Users\\David Mauder\\ray_results\\hc svm C\\svm_acc_c_0.92\\model_object.model",
            "C:\\Users\\David Mauder\\ray_results\\hc fcnn A\\fcnn_acc_a_0.735\\model_object.model",
            "C:\\Users\\David Mauder\\ray_results\\hc dm T\\dm_acc_t_0.59\\model_object.model",
            "C:\\Users\\David Mauder\\ray_results\\hc fm.fcnn.svm H\\dm_acc_h_0.58\\model_object.model",
        ],
        classifier_type="LCL",
        classification_cutoff=0,
    )
    hc_lcl.get_data(random_seed=42)
    hc_lcl.predict_lcl()


if __name__ == "__main__":
    main()
