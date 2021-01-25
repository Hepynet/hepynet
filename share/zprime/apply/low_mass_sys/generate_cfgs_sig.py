temp = '''config:
    include:
        - "share/zprime/train/low_mass_all_mass.yaml"

job:
    job_name: "apply-all-mass-sys-sig"
    job_type: "apply"
    load_job_name: "train-all-mass"

input:
    reset_feature: false
    # only remove negative events for training
    rm_negative_weight_events: false
    # apply to sys trees
    arr_path: "zprime/arrays"
    arr_version: "21-0115-sys"
    variation: "{p_variation}"
    campaign: "run2"
    channel: "dummy_channel"
    sig_list:
        - "sig_Zp005"
        - "sig_Zp007"
        - "sig_Zp009"
        - "sig_Zp011"
        - "sig_Zp013"
        - "sig_Zp015"
        - "sig_Zp017"
        - "sig_Zp019"
        - "sig_Zp023"
        - "sig_Zp027"
        - "sig_Zp031"
        - "sig_Zp035"
        - "sig_Zp039"
    bkg_list: []
    selected_features:
        - "mz2"
        - "ptl1"
        - "ptl2"
        - "ptl3"
        - "ptl4"
        - "etal1"
        - "etal2"
        - "etal3"
        - "etal4"
        - "mz1_mz2"
        - "ptz1"
        - "ptz2"
        - "mzz"
        - "ptzz"
        - "deltarl12"
        - "deltarl34"
        - "detal12"
        - "detal34"
    validation_features:
        - "mz1"
        - "mz2"
    feature_norm_alias:
        mz2: "mz2_p"

apply:
    book_fit_npy: true
    cfg_fit_npy:
        fit_npy_region:
        fit_npy_branches:
            - "mz1"
            - "mz2"
        npy_save_dir: "zprime/arrays_fit/21-0115-sys/low_mass/{p_variation}"

'''

sig_ntuple_names = [
    "tree_NOMINAL",
    "tree_EG_RESOLUTION_ALL__1down",
    "tree_EG_RESOLUTION_ALL__1up",
    "tree_EG_SCALE_AF2__1down",
    "tree_EG_SCALE_AF2__1up",
    "tree_EG_SCALE_ALL__1down",
    "tree_EG_SCALE_ALL__1up",
    "tree_EL_EFF_ID_CorrUncertaintyNP0__1down",
    "tree_EL_EFF_ID_CorrUncertaintyNP0__1up",
    "tree_EL_EFF_ID_CorrUncertaintyNP1__1down",
    "tree_EL_EFF_ID_CorrUncertaintyNP1__1up",
    "tree_EL_EFF_ID_CorrUncertaintyNP10__1down",
    "tree_EL_EFF_ID_CorrUncertaintyNP10__1up",
    "tree_EL_EFF_ID_CorrUncertaintyNP11__1down",
    "tree_EL_EFF_ID_CorrUncertaintyNP11__1up",
    "tree_EL_EFF_ID_CorrUncertaintyNP12__1down",
    "tree_EL_EFF_ID_CorrUncertaintyNP12__1up",
    "tree_EL_EFF_ID_CorrUncertaintyNP13__1down",
    "tree_EL_EFF_ID_CorrUncertaintyNP13__1up",
    "tree_EL_EFF_ID_CorrUncertaintyNP14__1down",
    "tree_EL_EFF_ID_CorrUncertaintyNP14__1up",
    "tree_EL_EFF_ID_CorrUncertaintyNP15__1down",
    "tree_EL_EFF_ID_CorrUncertaintyNP15__1up",
    "tree_EL_EFF_ID_CorrUncertaintyNP2__1down",
    "tree_EL_EFF_ID_CorrUncertaintyNP2__1up",
    "tree_EL_EFF_ID_CorrUncertaintyNP3__1down",
    "tree_EL_EFF_ID_CorrUncertaintyNP3__1up",
    "tree_EL_EFF_ID_CorrUncertaintyNP4__1down",
    "tree_EL_EFF_ID_CorrUncertaintyNP4__1up",
    "tree_EL_EFF_ID_CorrUncertaintyNP5__1down",
    "tree_EL_EFF_ID_CorrUncertaintyNP5__1up",
    "tree_EL_EFF_ID_CorrUncertaintyNP6__1down",
    "tree_EL_EFF_ID_CorrUncertaintyNP6__1up",
    "tree_EL_EFF_ID_CorrUncertaintyNP7__1down",
    "tree_EL_EFF_ID_CorrUncertaintyNP7__1up",
    "tree_EL_EFF_ID_CorrUncertaintyNP8__1down",
    "tree_EL_EFF_ID_CorrUncertaintyNP8__1up",
    "tree_EL_EFF_ID_CorrUncertaintyNP9__1down",
    "tree_EL_EFF_ID_CorrUncertaintyNP9__1up",
    "tree_EL_EFF_ID_SIMPLIFIED_UncorrUncertaintyNP0__1down",
    "tree_EL_EFF_ID_SIMPLIFIED_UncorrUncertaintyNP0__1up",
    "tree_EL_EFF_ID_SIMPLIFIED_UncorrUncertaintyNP1__1down",
    "tree_EL_EFF_ID_SIMPLIFIED_UncorrUncertaintyNP1__1up",
    "tree_EL_EFF_ID_SIMPLIFIED_UncorrUncertaintyNP10__1down",
    "tree_EL_EFF_ID_SIMPLIFIED_UncorrUncertaintyNP10__1up",
    "tree_EL_EFF_ID_SIMPLIFIED_UncorrUncertaintyNP11__1down",
    "tree_EL_EFF_ID_SIMPLIFIED_UncorrUncertaintyNP11__1up",
    "tree_EL_EFF_ID_SIMPLIFIED_UncorrUncertaintyNP12__1down",
    "tree_EL_EFF_ID_SIMPLIFIED_UncorrUncertaintyNP12__1up",
    "tree_EL_EFF_ID_SIMPLIFIED_UncorrUncertaintyNP13__1down",
    "tree_EL_EFF_ID_SIMPLIFIED_UncorrUncertaintyNP13__1up",
    "tree_EL_EFF_ID_SIMPLIFIED_UncorrUncertaintyNP14__1down",
    "tree_EL_EFF_ID_SIMPLIFIED_UncorrUncertaintyNP14__1up",
    "tree_EL_EFF_ID_SIMPLIFIED_UncorrUncertaintyNP15__1down",
    "tree_EL_EFF_ID_SIMPLIFIED_UncorrUncertaintyNP15__1up",
    "tree_EL_EFF_ID_SIMPLIFIED_UncorrUncertaintyNP16__1down",
    "tree_EL_EFF_ID_SIMPLIFIED_UncorrUncertaintyNP16__1up",
    "tree_EL_EFF_ID_SIMPLIFIED_UncorrUncertaintyNP17__1down",
    "tree_EL_EFF_ID_SIMPLIFIED_UncorrUncertaintyNP17__1up",
    "tree_EL_EFF_ID_SIMPLIFIED_UncorrUncertaintyNP2__1down",
    "tree_EL_EFF_ID_SIMPLIFIED_UncorrUncertaintyNP2__1up",
    "tree_EL_EFF_ID_SIMPLIFIED_UncorrUncertaintyNP3__1down",
    "tree_EL_EFF_ID_SIMPLIFIED_UncorrUncertaintyNP3__1up",
    "tree_EL_EFF_ID_SIMPLIFIED_UncorrUncertaintyNP4__1down",
    "tree_EL_EFF_ID_SIMPLIFIED_UncorrUncertaintyNP4__1up",
    "tree_EL_EFF_ID_SIMPLIFIED_UncorrUncertaintyNP5__1down",
    "tree_EL_EFF_ID_SIMPLIFIED_UncorrUncertaintyNP5__1up",
    "tree_EL_EFF_ID_SIMPLIFIED_UncorrUncertaintyNP6__1down",
    "tree_EL_EFF_ID_SIMPLIFIED_UncorrUncertaintyNP6__1up",
    "tree_EL_EFF_ID_SIMPLIFIED_UncorrUncertaintyNP7__1down",
    "tree_EL_EFF_ID_SIMPLIFIED_UncorrUncertaintyNP7__1up",
    "tree_EL_EFF_ID_SIMPLIFIED_UncorrUncertaintyNP8__1down",
    "tree_EL_EFF_ID_SIMPLIFIED_UncorrUncertaintyNP8__1up",
    "tree_EL_EFF_ID_SIMPLIFIED_UncorrUncertaintyNP9__1down",
    "tree_EL_EFF_ID_SIMPLIFIED_UncorrUncertaintyNP9__1up",
    "tree_EL_EFF_Iso_TOTAL_1NPCOR_PLUS_UNCOR__1down",
    "tree_EL_EFF_Iso_TOTAL_1NPCOR_PLUS_UNCOR__1up",
    "tree_EL_EFF_Reco_TOTAL_1NPCOR_PLUS_UNCOR__1down",
    "tree_EL_EFF_Reco_TOTAL_1NPCOR_PLUS_UNCOR__1up",
    "tree_FT_EFF_B_systematics__1down",
    "tree_FT_EFF_B_systematics__1up",
    "tree_FT_EFF_C_systematics__1down",
    "tree_FT_EFF_C_systematics__1up",
    "tree_FT_EFF_Light_systematics__1down",
    "tree_FT_EFF_Light_systematics__1up",
    "tree_FT_EFF_extrapolation__1down",
    "tree_FT_EFF_extrapolation__1up",
    "tree_FT_EFF_extrapolation_from_charm__1down",
    "tree_FT_EFF_extrapolation_from_charm__1up",
    "tree_JET_BJES_Response__1up",
    "tree_JET_BJES_Response__1down",
    "tree_JET_EffectiveNP_Detector1__1up",
    "tree_JET_EffectiveNP_Detector1__1down",
    "tree_JET_EffectiveNP_Detector2__1up",
    "tree_JET_EffectiveNP_Detector2__1down",
    "tree_JET_EffectiveNP_Mixed1__1up",
    "tree_JET_EffectiveNP_Mixed1__1down",
    "tree_JET_EffectiveNP_Mixed2__1up",
    "tree_JET_EffectiveNP_Mixed2__1down",
    "tree_JET_EffectiveNP_Mixed3__1up",
    "tree_JET_EffectiveNP_Mixed3__1down",
    "tree_JET_EffectiveNP_Modelling1__1up",
    "tree_JET_EffectiveNP_Modelling1__1down",
    "tree_JET_EffectiveNP_Modelling2__1up",
    "tree_JET_EffectiveNP_Modelling2__1down",
    "tree_JET_EffectiveNP_Modelling3__1up",
    "tree_JET_EffectiveNP_Modelling3__1down",
    "tree_JET_EffectiveNP_Modelling4__1up",
    "tree_JET_EffectiveNP_Modelling4__1down",
    "tree_JET_EffectiveNP_Statistical1__1up",
    "tree_JET_EffectiveNP_Statistical1__1down",
    "tree_JET_EffectiveNP_Statistical2__1up",
    "tree_JET_EffectiveNP_Statistical2__1down",
    "tree_JET_EffectiveNP_Statistical3__1up",
    "tree_JET_EffectiveNP_Statistical3__1down",
    "tree_JET_EffectiveNP_Statistical4__1up",
    "tree_JET_EffectiveNP_Statistical4__1down",
    "tree_JET_EffectiveNP_Statistical5__1up",
    "tree_JET_EffectiveNP_Statistical5__1down",
    "tree_JET_EffectiveNP_Statistical6__1up",
    "tree_JET_EffectiveNP_Statistical6__1down",
    "tree_JET_EtaIntercalibration_Modelling__1up",
    "tree_JET_EtaIntercalibration_Modelling__1down",
    "tree_JET_EtaIntercalibration_NonClosure_2018data__1up",
    "tree_JET_EtaIntercalibration_NonClosure_2018data__1down",
    "tree_JET_EtaIntercalibration_NonClosure_highE__1up",
    "tree_JET_EtaIntercalibration_NonClosure_highE__1down",
    "tree_JET_EtaIntercalibration_NonClosure_negEta__1up",
    "tree_JET_EtaIntercalibration_NonClosure_negEta__1down",
    "tree_JET_EtaIntercalibration_NonClosure_posEta__1up",
    "tree_JET_EtaIntercalibration_NonClosure_posEta__1down",
    "tree_JET_EtaIntercalibration_TotalStat__1up",
    "tree_JET_EtaIntercalibration_TotalStat__1down",
    "tree_JET_Flavor_Composition__1up",
    "tree_JET_Flavor_Composition__1down",
    "tree_JET_Flavor_Response__1up",
    "tree_JET_Flavor_Response__1down",
    "tree_JET_JER_DataVsMC_MC16__1up",
    "tree_JET_JER_DataVsMC_MC16__1down",
    "tree_JET_JER_EffectiveNP_1__1up",
    "tree_JET_JER_EffectiveNP_1__1down",
    "tree_JET_JER_EffectiveNP_2__1up",
    "tree_JET_JER_EffectiveNP_2__1down",
    "tree_JET_JER_EffectiveNP_3__1up",
    "tree_JET_JER_EffectiveNP_3__1down",
    "tree_JET_JER_EffectiveNP_4__1up",
    "tree_JET_JER_EffectiveNP_4__1down",
    "tree_JET_JER_EffectiveNP_5__1up",
    "tree_JET_JER_EffectiveNP_5__1down",
    "tree_JET_JER_EffectiveNP_6__1up",
    "tree_JET_JER_EffectiveNP_6__1down",
    "tree_JET_JER_EffectiveNP_7restTerm__1up",
    "tree_JET_JER_EffectiveNP_7restTerm__1down",
    "tree_JET_JvtEfficiency__1down",
    "tree_JET_JvtEfficiency__1up",
    "tree_JET_Pileup_OffsetMu__1up",
    "tree_JET_Pileup_OffsetMu__1down",
    "tree_JET_Pileup_OffsetNPV__1up",
    "tree_JET_Pileup_OffsetNPV__1down",
    "tree_JET_Pileup_PtTerm__1up",
    "tree_JET_Pileup_PtTerm__1down",
    "tree_JET_Pileup_RhoTopology__1up",
    "tree_JET_Pileup_RhoTopology__1down",
    "tree_JET_PunchThrough_MC16__1up",
    "tree_JET_PunchThrough_MC16__1down",
    "tree_JET_SingleParticle_HighPt__1up",
    "tree_JET_SingleParticle_HighPt__1down",
    "tree_MUON_EFF_ISO_STAT__1down",
    "tree_MUON_EFF_ISO_STAT__1up",
    "tree_MUON_EFF_ISO_SYS__1down",
    "tree_MUON_EFF_ISO_SYS__1up",
    "tree_MUON_EFF_RECO_STAT__1down",
    "tree_MUON_EFF_RECO_STAT__1up",
    "tree_MUON_EFF_RECO_STAT_LOWPT__1down",
    "tree_MUON_EFF_RECO_STAT_LOWPT__1up",
    "tree_MUON_EFF_RECO_SYS__1down",
    "tree_MUON_EFF_RECO_SYS__1up",
    "tree_MUON_EFF_RECO_SYS_LOWPT__1down",
    "tree_MUON_EFF_RECO_SYS_LOWPT__1up",
    "tree_MUON_EFF_TTVA_STAT__1down",
    "tree_MUON_EFF_TTVA_STAT__1up",
    "tree_MUON_EFF_TTVA_SYS__1down",
    "tree_MUON_EFF_TTVA_SYS__1up",
    "tree_MUON_ID__1down",
    "tree_MUON_ID__1up",
    "tree_MUON_MS__1down",
    "tree_MUON_MS__1up",
    "tree_MUON_SAGITTA_RESBIAS__1down",
    "tree_MUON_SAGITTA_RESBIAS__1up",
    "tree_MUON_SAGITTA_RHO__1down",
    "tree_MUON_SAGITTA_RHO__1up",
    "tree_MUON_SCALE__1down",
    "tree_MUON_SCALE__1up",
    "tree_PRW_DATASF__1down",
    "tree_PRW_DATASF__1up",
]

for variation in sig_ntuple_names:
    with open(f"low_mass_all-mass_apply_sys_sig_{variation}.yaml", "w+") as file:
        file.write(temp.format(p_variation=variation))

