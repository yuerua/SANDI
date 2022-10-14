def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='#'): #chr(0x00A3)
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration / total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end="")
    # Print New Line on Complete
    if iteration == total:
        print()

def printProgressBar_keras(iteration, total, prefix='', suffix='', length=100, fill='='):
    filledLength = int(length * iteration / total)
    bar = fill * filledLength + '>' + '.' * (length - filledLength)
    print('\r %s [%s] %s' % (prefix, bar, suffix), end="")
    # Print New Line on Complete
    if iteration == total:
        print()

# if __name__ == '__main__':
#     printProgressBar(0, len(patients), prefix='Progress:', suffix='Complete', length=50)

#     printProgressBar(num_patients, len(patients), prefix='Progress:',
#                       suffix='Completed for %s'%patient_name, length=50)
#
#     printProgressBar_keras(index + 1,
#                      no_batch,
#                      prefix='%i/%i' % ((index + 1) * batch_size, train_data_a.shape[0]),
#                      suffix='- ETA: %2is - loss: %.4f - acc: %.4f' % (time_per_ite * no_batch, loss, acc),
#                      length=30)