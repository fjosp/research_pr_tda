df = pd.read_csv('aa.csv', sep=',')
df['purchase_date'] = pd.to_datetime(df.purchase_date)


grouped = df.groupby('card_id')
n = len(grouped)

f, axs = plt.subplots(nrows=n, ncols=1, sharex=False, sharey=False, figsize=(15,15))



counter = 0
for name, group in grouped:
#     print(name)
    group = group.sort_values(by = 'purchase_date')
    axs[counter].plot(group['purchase_date'], group['purchase_amount'], '-o', c='c', 
                      markerfacecolor='m', mec = 'm', markersize=4)
    axs[counter].set_title(name)
    axs[counter].grid(color='grey', linestyle='-', linewidth=0.3)
    
    
    counter += 1
