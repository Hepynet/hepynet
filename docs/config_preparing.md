# Prepare a config file

## General

This framework use an ".yaml" file to set up the inputs, model structure, output
format and so on.

There 6 basic sections in a configuration file:

- **config**: specify other configuration files to be included
- **job**: job information
- **input**: config inputs
- **train**: set up model training
- **apply**: set up model applying

## **[config]** Higher level configs to be included

- **include [list]**: Path to other yaml config files. Recursive include is permitted.
  If the included file have same setting entry as current one, the current
  setting will override the included settings

## **[job]** Job setup

- **job_name [str]**: Job identifier.
- **job_type [str]**: Set to "train", if need to train a new model. Set to "apply", if
  need to apply existing model to new samples.
- **load_job_name [str]**: Job to be loaded for "apply" job.
- **save_dir [str]**: Path to the directory where all outputs will be saved. By default, "apply" job results will be saved in the same directory as "train" job.

## **[input]** Inputs setup

- **arr_path [str]**: Relative path to data_path defined in [pc_meta.yaml](/share/cross_platform/pc_meta.yaml)
- **arr_version [str]**: Which version of input arrays to use. e.g. v08, rel_125
- **campaign [str]**: Campaign to use. e.g. mc16a, run2, all
- **region [str]**: Signal/control region. If no region is defined in array folder, set this to **""**
- **channel [str]**: Channel to be used. (if no channel or only one channel used in the analysis, please create a dummy channel when preparing the input arrays)
- **sig_key [str]**: Name of signal sample to be used. e.g. sig_500GeV, all_norm
- **sig_list [list]**: All signal samples used.
- **bkg_key [str]**: Name of background sample to be used. e.g. bkg_qcd, all
- **bkg_list [list]**: All background samples used.
- **data_key [str]**: Name of data sample to be used.
- **data_list [list]**: All data samples used.
- **selected_features [list]**: Features used for training.
- **selected_features [list]**: Features used for validation.
- **cut_features [list]**: Features used for cutting input. e.g. ["pt", "paired"]
- **cut_values [list]**: Cut thresholds. e.g. [50, 1]
- **cut_type [list]**: Type of cuts. e.g. [">", "="]

- **norm_array [bool]**: Whether to perform normalization for input feature
  distributions. Arrays will be normalized by: (x - mean) / sigma
- **sig_sumofweight [num]**: Total weight for signal events to be normalized to.
- **bkg_sumofweight [num]**: Total weight for background events to be normalized to.
- **data_sumofweight [num]**: Total weight for data events to be normalized to.
- **reset_feature [bool]**: Whether to perform feature reset. (for pDNN)
- **reset_feature_name [str]**: The background reset_feature will be randomly reset
  but has same distribution as signal reset_feature. (for pDNN)
- **rm_negative_weight_events [bool]**: Whether to use negative events.
- **rdm_seed [int]**: random seed when random numbers need to be generated.

## **[train]** DNN training specifications

- **model_name [str]**: Identifier for current model.
- **model_class [str]**: Model class name to be used for current model. e.g. Model_Sequential_Flat
- **output_bkg_node_names [list]**: For multi-nodes training
- **layers [int]**: Number of layers.
- **nodes [int]**: Number of nodes per layer.
- **learn_rate [num]**: Learning rate.
- **learn_rate_decay [num]**: Learning rate decay rate.
- **batch_size [int]**: Batch size for each propagation.
- **epochs [int]**: Maximum epochs for fitting process.
- **momentum [num]**: Learning momentum.
- **nesterov [bool]**: Whether to use nesterov momentum.
- **dropout_rate [num]**: Dropout rate per layer.
- **sig_class_weight [num]**: Signal class weight. Training will pay more attention to
  signal samples if sig_class_weight is higher compared to other classes.
- **bkg_class_weight [num]**: Background class weight. Training will pay more attention
  to background samples if bkg_class_weight is higher
- **test_rate [num]**: Ratio of standalone samples for testing purpose.
- **val_split [num]**: Ratio of validation samples for validation purpose. For example:
  if test\*rate is 0.2 and val_split is 0.25, then test/train/validation ratio
  will be 0.2 / (1 - 0.2) x (1 - 0.25) = 0.6 / (1 - 0.2) \_ 0.25 = 0.2

- **use_early_stop [bool]**: Whether to use early stop method.
- **early_stop_paras [dict]**: Setup early-stopping.
  - **monitor [str]**: The value for early stop monitoring.
  - **min_delta [num]**: The minimum difference for early stopping.
  - **patience [int]**: Early stop checking interval.
  - **mode [str]**: The way to judge early stopping.
  - **restore_best_weights [bool]**: Whether to restore the best model weights.
- **train_metrics [list]**: Unweighted training metrics for performance monitoring.
- **train_metrics_weighted [list]**: Weighted training metrics for performance
  monitoring.
- **save_model [bool]**: Whether to save the model.
- **verbose [int]**: Verbose level of output information.

## **[apply]** DNN applying specifications

- **plot_bkg_list**: Stack order for background samples. If empty, the background
  samples will be stacked from small to large total weights.
- **plot_density**: Whether to plot normalized result.
- **apply_data**: Whether to plot data as well.
- **apply_data_range**: Range to plot data points in DNN scores plot.
- **kine_cfg**: Style config for kinematic plots.
- **show_report**: Whether to show the report during the plotting.
- **save_pdf_report**: Whether to save pdf report.
- **save_tb_logs**: Whether to save TensorBoard logs.
- **verbose**: Verbose mode in Keras.

- **book_history [bool]**: Make metrics history plots.
- **cfg_history [dict]**: Customized history plots. e.g. to customize accuracy curve you can set following:

  ```yaml
  cfg_history:
    accuracy:
      plot_title: "accuracy history"
      save_format: "png"
  ```

- **book_roc [bool]**: Make ROC curves.
- **book_mva_scores_data_mc [bool]**: Make scores distributions for different samples
- **cfg_mva_scores_data_mc [dict]**:
  - **sig_list [list]**: Signal samples to be included.
  - **bkg_list [list]**: Background samples to be included.
  - **apply_data [bool]**: Whether to show data.
  - **apply_data_range [list]**: Range of data to be shown. e.g. [50, 1000]
  - **plot_title [str]**
  - **bins [int]**: Number of bins.
  - **range [list]**: Plot range.
  - **density [bool]**: Whether normalized to unit.
  - **log [bool]**: Log scale.
  - **save_format [str]**: e.g. png, jpg
  - **use_root [bool]**: [Not recommended] Use ROOT for plotting (should setup root in the environment)
- **book_train_test_compare [bool]**: Compare scores of train/test dataset.
- **cfg_train_test_compare [dict]**:
  - **sig_key [str]**
  - **bkg_key [str]**
  - **plot_title [str]**
  - **bins [int]**: Number of bins.
  - **range [list]**: Plot range.
  - **density [bool]**: Whether normalized to unit.
  - **log [bool]**: Log scale.
  - **save_format [str]**: e.g. png, jpg
- **book_kine_study [bool]**: Make input kinematic distributions.
- **book_cut_kine_study [bool]**: Make input kinematic distributions with DNN cuts.
- **cfg_kine_study**
  - **bins [int]**: Number of bins.
  - **range [list]**: Plot range.
  - **histtype [str]**: Set matplotlib histtype
  - **alpha [num]**: Set matplotlib alpha
  - **density [bool]**: Whether normalized to unit.
  - **sig_color [str]**: Set matplotlib color
  - **bkg_color [str]**: Set matplotlib color
  - **dnn_cut_list [list]**: List of DNN cuts to check.
  - **save_format [str]**: e.g. png, jpg
- **book_cor_matrix [bool]**: Makes correlation table for inputs.
- **book_importance_study [bool]**: Study importance of input features.
- **book_significance_scan [bool]**: Scan significance vs DNN cut.
- **cfg_significance_scan [dict]**:
  - **significance_algo [str]**: Algorithm to check significance, could be: asimov, s_b, s_sqrt_b, s_sqrt_sb, asimov_rel, s_b_rel, s_sqrt_b_rel, s_sqrt_sb_rel
- **book_fit_npy [bool]**: Generate numpy arrays for fitting. (Can be transformed to root file with hepynet_root_npy)
- **cfg_fit_npy [dict]**:
  - **fit_npy_region [str]**: Signal/control region of fitting.
  - **fit_npy_branches [str]**: Features included in fitting.
  - **npy_save_dir [str]**: Directory path to save output arrays

**Note**:

- Default settings are defined in [hepynet/common/config_default.py](hepynet/common/config_default.py)
- If a setting is not needed, comment it but don't leave blank. "**blank**" will be interpreted as **None** which will result in different behavior for some utilities.
