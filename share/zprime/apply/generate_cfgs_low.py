#temp = """config:
#    include:
#        - "share/zprime/apply/low_mass_all-mass_apply.yaml"
#
#job:
#    job_name: "apply-{p_mass}"
#    load_job_name: "train-all-mass"
#
#input:
#    sig_key: "sig_Zp{p_mass:03d}"
#    cut_features:
#        - "quadtype"
#        - "mz2"
#        - "mz2"
#    cut_values:
#        - 2
#        - {p_mass_min}
#        - {p_mass_max}
#    cut_types:
#        - "="
#        - ">"
#        - "<"
#
#"""

temp = '''config:
    include:
        - "share/zprime/apply/low_mass_no-19_apply.yaml"

job:
    job_name: "apply-{p_mass}"
    load_job_name: "train-no-19"

input:
    sig_key: "sig_Zp{p_mass:03d}"
    cut_features:
        - "quadtype"
        - "mz2"
        - "mz2"
    cut_values:
        - 2
        - {p_mass_min}
        - {p_mass_max}
    cut_types:
        - "="
        - ">"
        - "<"

'''

for mass in [5, 7, 9, 11, 13, 15, 17, 19, 23, 27, 31, 35, 39]:
    sigma5 = 5 * (-0.0202966 + 0.0190822 * mass)
    #with open(f"low_mass_all-mass_apply_{mass}.yaml", "w+") as file:
    with open(f"low_mass_no-19_apply_{mass}.yaml", "w+") as file:
        file.write(
            temp.format(p_mass=mass, p_mass_min=mass - sigma5, p_mass_max=mass + sigma5)
        )

