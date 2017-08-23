import sys
sys.path.append("../tools/")
import loadEnron
import plot_data

plot = plot_data.plotVariables
data = loadEnron.load()

output = 'plots/%s-%s.png' % ('total_payments','total_stock_value')
plot(data, "total_payments", "total_stock_value", output, x_annotate=40000000)


data.pop('LAY KENNETH L', 0)

output2 = 'plots/%s-%s.png' % ('total_payments','total_stock_value' + "_exclude_lay")
plot(data, "total_payments", "total_stock_value", output2, x_annotate=10000000, y_annotate=10000000)

