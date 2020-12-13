#for mass in 5 7 9 11 13 15 17 19 23 27 31 35 39
#do
#    execute_pdnn_jobs share/zprime/apply/low_mass_all-mass_apply_$mass.yaml
#done

for mass in 5 7 9 11 13 15 17 19 23 27 31 35 39
do
    execute_pdnn_jobs share/zprime/apply/low_mass_no-19_apply_$mass.yaml
done