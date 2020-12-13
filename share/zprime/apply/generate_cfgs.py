#temp1 = '''config:
#    include:
#        - "share/zprime/apply/high_mass_apply.yaml"
#
#job:
#    job_name: "high-mass-{p_mass}-apply-{p_mass}"
#    load_job_name: "train-{p_mass}"
#
#apply:
#    cfg_fit_ntup:
#        fit_ntup_region:
#        fit_ntup_branches:
#            - "mz1"
#            - "mz2"
#        ntup_save_dir: "zprime/ntuples/fit_ntup_model_{p_mass}"
#
#'''
#
#temp2 = '''config:
#    include:
#        - "share/zprime/apply/high_mass_apply.yaml"
#
#job:
#    job_name: "apply-{p_mass}"
#    load_job_name: "train-all-mass"
#
#input:
#    sig_key: "sig_Zp{p_mass:03d}"
#
#apply:
#    cfg_fit_ntup:
#        fit_ntup_region:
#        fit_ntup_branches:
#            - "mz1"
#            - "mz2"
#        ntup_save_dir: "zprime/ntuples/fit_ntup_model_{p_mass}"
#
#'''

temp = '''config:
    include:
        - "share/zprime/apply/high_mass_no-69_apply.yaml"

job:
    job_name: "apply-{p_mass}"
    load_job_name: "train-no-69"

input:
    sig_key: "sig_Zp{p_mass:03d}"

apply:
    cfg_fit_ntup:
        fit_ntup_region:
        fit_ntup_branches:
            - "mz1"
            - "mz2"
        ntup_save_dir: "zprime/ntuples/fit_ntup_model_{p_mass}"

'''


for mass in [42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72, 75]:
    with open(f"high_mass_no-69_apply_{mass}.yaml", "w+") as file:
        file.write(temp.format(p_mass = mass))
    