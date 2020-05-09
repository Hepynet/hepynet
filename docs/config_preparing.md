# Prepare a config file

## General
This framework use an ".ini" file to set up the inputs, model structure, output 
format and so on.

There 6 basic sections in a configuration file:
* config: specify other configuration files to be included
* job: set meta data of the job
* array: specify input arrays path and configurations
* model: set up model structure and hyper-parameters
* para_scan: set up hyper-parameters tuning 
* report: define output report format

Based on the example file "example_cfg.ini" in the same directory, instructions
are given on how to prepare the configuration in each section.

## **[config]**
* **include**: Path to the ini_file. The included file can include another file.
    If the included file have same setting entry as current one, the current
    setting will override the included settings

## **[job]**
* **job_name**: A brief string as job identifier.
* **job_type**: Set to "train", if need to train a new model. Set to "apply", if
    need to apply existing model to new samples.
* **save_dir**: Path to the directory where all outputs will be saved.

## **[array]**
Please make sure the input array are sorted in the following manner:
/arr_directory/arr_version/campaign/
* **arr_version**: Which version of input arrays to use.
* **campaign**: Campaign name(defined in your array preparation process). For
    example: mc16a, run2, all
* **bkg_dict_path**: Path to the background arrays' directory.
* **bkg_key**: Which background components to be used. For example: diboson,
    top, all(defined in your array preparation process)
* **bkg_sumofweight**: Total weight for background events to be normalized to
    for training. 
* **sig_dict_path**: Path to the signal arrays' directory.
* **sig_key**: Which signal components to be used. For example: 500GeV
* **sig_sumofweight**: Total weight for signal events to be normalized to for
    training.
* **data_dict_path**: Path to the data arrays' directory.
* **data_key**: Which data components to be used.
* **data_sumofweight**: Total weight for data events to be normalized to.
* **channel**: Physic channel of the analysis.(Should create a branch of array
    to specify channel, if no channel or only one channel used in the analysis,
    please create a dummy channel.) For example: emu, sf
* **norm_array**: Whether to perform normalization for input feature
    distributions. Array will be normalized by: (x - mean) / sigma
* **bkg_list**: List of background components. (should be consist with array
    preparation process)
* **sig_list**: List of signal components. (should be consist with array
    preparation process)
* **data_list**: List of data components.
* **selected_features**: List of input features used for training.
* **reset_feature**: Whether to perform feature reset.
* **reset_feature_name**: The background reset_feature will be randomly reset
    but has same distribution as signal reset_feature. This is for pDNN studies.

## **[model]**

* **model_name**: Identifier for current model.
* **model_class**: Class name to be used for current model. For example: Model_Sequential_Flat
* **layers**: Number of layers.
* **nodes**: Number of nodes per layer.
* **dropout_rate**: Dropout rate per layer.
* **momentum**: Learning momentum.
* **nesterov**: Whether to use nesterov momentum.
* **rm_negative_weight_events**: Whether to use negative events for training.
* **test_rate**: Ratio of standalone samples for testing purpose.
* **val_split**: Ratio of validation samples for validation purpose. For example:
    if test_rate is 0.2 and val_split is 0.25, then test/train/validation ratio
    will be 0.2 / (1 - 0.2) * (1 - 0.25) = 0.6 / (1 - 0.2) * 0.25 = 0.2
* **learn_rate**: Learning rate during fitting.
* **learn_rate_decay**: Learning rate decay during fitting.
* **batch_size**: Batch size for each propagation.
* **epochs**: Maximum epochs for fitting process.
* **sig_class_weight**: Signal class weight. Training will pay more attention to
    signal samples if sig_class_weight is higher compared to other classes.
* **bkg_class_weight**: Background class weight. Training will pay more attention
    to background samples if bkg_class_weight is higher 
* **use_early_stop**: Whether to use early stop method.
* **early_stop_monitor**: The value for early stop monitoring.
* **early_stop_min_delta**: The minimum difference for early stopping.
* **early_stop_patience**: Early stop checking interval.
* **early_stop_mode**: The way to judge early stopping.
* **early_stop_restore_best_weights**: Whether to restore the best model weights.
* **train_metrics**: Unweighted training metrics for performance monitoring.
* **train_metrics_weighted**: Weighted training metrics for performance 
    monitoring.
* **save_model**: Whether to save the model.

## **[para_scan]**
* **perform_para_scan**: Whether to perform hyper-parameter scan.
* **max_scan_iterations**: Maximum scan iterations.
* **para_scan_cfg**: Path to detailed config for scan space.

## **[report]**

* **plot_bkg_list**: Stack order for background samples. If empty, the background
    samples will be stacked from small to large total weights.
* **plot_density**: Whether to plot normalized result.
* **apply_data**: Whether to plot data as well.
* **apply_data_range**: Range to plot data points in DNN scores plot.
* **kine_cfg**: Style config for kinematic plots.
* **show_report**: Whether to show the report during the plotting.
* **save_pdf_report**: Whether to save pdf report.
* **save_tb_logs**: Whether to save TensorBoard logs.
* **verbose**: Verbose mode in Keras.