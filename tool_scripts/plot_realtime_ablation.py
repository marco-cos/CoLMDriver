import matplotlib.pyplot as plt

realtime = [30.74, 43.33, 86.49, 48.35, 89.25, 91.65]
non_realtime = [37.24, 42.50, 82.18, 60.88, 96.18, 98.27]

baselines = ['VAD', 'UniAD', 'TCP', 'LMDrive', 'CoDriving', 'Ours']

x = range(len(baselines))
width = 0.35

fig, ax = plt.subplots(figsize=(11, 6))
plt.subplots_adjust(left=0.05, right=0.95)
rects2 = ax.bar([i - width/2 for i in x], non_realtime, width, label='Ideal(No latency)', color="#DEA79F")
rects1 = ax.bar([i + width/2 for i in x], realtime, width, label='Latency-aware', color='#A8B6F0')

ax.set_ylabel("Driving Score", fontsize=24)
ax.set_title('Effects of Inference Latency', fontsize=28)
ax.set_xticks(x)
ax.set_xticklabels(baselines, fontsize=20)

ax.yaxis.set_tick_params(labelsize=20)
ax.legend(fontsize=20)

# def autolabel(rects):
#     for rect in rects:
#         height = rect.get_height()
#         ax.annotate('{}'.format(height),
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),  # 3 points vertical offset
#                     textcoords="offset points",
#                     ha='center', va='bottom')

# autolabel(rects1)
# autolabel(rects2)
plt.tight_layout()

plt.savefig("realtime_ablation.png")
