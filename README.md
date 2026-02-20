# Assignment-Week-4
# Assuming 'waves_cleaned' contains the preprocessed waveforms and
# 'clusters_gmm' contains the predicted labels from the GMM model (0=sea ice, 1=lead)

# Calculate means
mean_ice = np.mean(waves_cleaned[clusters_gmm == 0], axis=0)
mean_lead = np.mean(waves_cleaned[clusters_gmm == 1], axis=0)

# Calculate standard deviations
std_ice = np.std(waves_cleaned[clusters_gmm == 0], axis=0)
std_lead = np.std(waves_cleaned[clusters_gmm == 1], axis=0)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(mean_ice, label='Sea Ice Mean', color='blue')
plt.fill_between(range(len(mean_ice)), mean_ice - std_ice, mean_ice + std_ice, alpha=0.3, color='blue', label='Sea Ice ±1σ')
plt.plot(mean_lead, label='Lead Mean', color='red')
plt.fill_between(range(len(mean_lead)), mean_lead - std_lead, mean_lead + std_lead, alpha=0.3, color='red', label='Lead ±1σ')
plt.title('Mean and Standard Deviation of Echoes for Each Class')
plt.xlabel('Range Bin')
plt.ylabel('Power')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
