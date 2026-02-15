#%% imports
from functions import *
from utils.io import *

#%% hyperparams
fs_res = 50 # resampling frequency (Hz)
n_epochs = 10
maxlag = 4
alpha = 0.05

#%% load and segment
subj_id = 'sub-017'
filepath = f"../dataset/derivatives/{subj_id}/eeg/{subj_id}_task-eyesclosed_eeg.set"
eeg, fs, channels = load_eeg(filepath, resample=fs_res, preload=True)

epochs = split_epochs(eeg, n_epochs=n_epochs) # split into 10 equal segments
print(epochs.shape)

all_subs_report = load_data("results/20260128_110832_allsubs_stationarity/data/saved_data.pkl")
sub = all_subs_report[subj_id] # to skip unnecessary stationarity checks

#%% granger 
gran_pvals = granger_ecn(epochs, channels, maxlag=4, current_subject=sub, ic=True)

#%% lingam
#ling_strength, ling_pvals = lingam_ecn(epochs, channels, maxlag, current_subject=sub)

# bootstrap
ling_strength, ling_probs = lingam_ecn_boot(epochs, channels, maxlag, current_subject=sub)

# jackknife
# ling_strength = lingam_ecn(epochs, channels, maxlag, current_subject=sub)
# ling_std = lingam_ecn_jk(epochs, channels, maxlag, current_subject=sub)

# no lags
#ling_strength, ling_bin_adj = lingam_ecn_no_lags(epochs, channels)

#%% load results
#channels, gran_pvals, gran_strength, gran_bin_adj, ling_strength, ling_bin_adj = load_data("results/20260121_162827_no_lags_ling/data/saved_data.pkl")
#channels, gran_pvals, gran_strength, gran_bin_adj, _, _ = load_data("results/20260121_162827_no_lags_ling/data/saved_data.pkl")

#%% plot ECNs
#plot_ecn(gran_strength, title="Granger ECN", threshold=2.0)  # p < 0.01
plot_ecn(ling_strength, title="LiNGAM ECN", threshold=1)

#%% save results
# res_granger = {"gran_pvals":gran_pvals, "gran_strength":gran_strength, "gran_bin_adj":gran_bin_adj}
# res_lingam = {"ling_strength":ling_strength, "ling_bin_adj":ling_bin_adj}
# res={"channels":channels, "granger":res_granger, "lingam":res_lingam}
# save_results(res)
print('the end')