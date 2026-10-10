import numpy as np
import matplotlib.pyplot as plt

predicted_value1 = np.loadtxt('K_25.txt')
#predicted_value1 = np.log(predicted_value1)

actual_value = np.loadtxt('ref')
actual_value = np.exp(actual_value)
mse = np.mean((predicted_value1 - actual_value)**2)
rmse1 = np.sqrt(mse)

print("Root Mean Squared Error1: ", rmse1)
#spread = np.max(np.abs(predicted_value))
spread1 = np.var(predicted_value1)
print('spread1:',spread1)


predicted_value2 = np.loadtxt('K_9_1.txt')
#predicted_value2 = np.log(predicted_value2)
#actual_value = np.exp(actual_value)
mse1 = np.mean((predicted_value2 - actual_value)**2)
rmse2 = np.sqrt(mse1)

print("Root Mean Squared Error2: ", rmse2)
#spread = np.max(np.abs(predicted_value))
spread2 = np.var(predicted_value2)
print('spread2:',spread2)


predicted_value3 = np.loadtxt('K_9_2.txt')
#predicted_value3 = np.log(predicted_value3)
#actual_value = np.exp(actual_value)
mse2 = np.mean((predicted_value3 - actual_value)**2)
rmse3 = np.sqrt(mse2)

print("Root Mean Squared Error3: ", rmse3)
#spread = np.max(np.abs(predicted_value))
spread3 = np.var(predicted_value3)
print('spread3:',spread3)

x_values = ['Scenario 1', 'Scenario 2', 'Scenario 3']
#y_values = [1.020879843163958, 0.9947052480323181,0.8722251990720523]
#z_values = [0.8983010653555958, 0.7945609677343268, 0.7708928847627357]
y_values = [rmse3, rmse2, rmse1]
z_values = [spread3, spread2, spread1]

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
# Create a scatter plot
lns1 = ax.plot(x_values, y_values, marker = 'o', color='red', linestyle='-', label='RMSE')
ax2 = ax.twinx()
lns2 = ax2.plot(x_values, z_values, marker = 's', color='black', linestyle= '-', label='Spread')
lns = lns1 + lns2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=0)

#plt.legend(loc='upper right', fontsize='medium')
#plt.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize='medium', ncol=1)
#plt.legend(loc='upper right')
#ax2 = ax1.twinx()
#ax2.plot(x_values, z_values, marker = 's', color='black', linestyle= '-', label='Spread')
ax.set_ylabel('RMSE')
ax2.set_ylabel('Spread')
ax.set_ylim(0,5)
ax2.set_ylim(1, 19)

#ax2.legend(loc='right')
#plt.legend(loc='upper right', fontsize='medium')
#plt.legend(loc='upper right', bbox_to_anchor=(1, 0.92), fontsize='medium', ncol=1)
# Set the title and axis labels
#plt.title('Scatter plot with categorical variable on the horizontal axis')
#plt.xlabel('Categories')
#plt.ylabel('Values')
plt.savefig('spd')
# Show the plot
plt.show()