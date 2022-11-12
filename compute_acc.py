

accuracy = {}
acc_nums = {}
with open('modelarts-job-c0a748c1-47a8-45d0-9603-2097883482c0-worker-0.log') as f:
    lines = f.readlines()
    for line in lines:
        if 'Validation-Loss' in line:
            contents = line.strip().split(' ')
            ckpt_index = int(contents[1].strip(','))
            if str(ckpt_index) not in accuracy.keys():
                acc_nums[str(ckpt_index)] = 1
                accuracy[str(ckpt_index)] = float(contents[8].strip(','))
            else:
                acc_nums[str(ckpt_index)] += 1
                accuracy[str(ckpt_index)] += float(contents[8].strip(','))

print(accuracy)
print(acc_nums)

mean_acc = []
acc_go = acc_nums.keys()
acc_lo = accuracy.keys()
for key in acc_lo:
    if key not in acc_go:
        print('Wrong key!!!!!!!')
    else:
        mean_acc.append(accuracy[key]/acc_nums[key])

print(mean_acc)
print(max(mean_acc))
