import numpy as np
import math
import NN_From_Scratch as NN

save_path = ""  # 保存模型路径

# 字符串转浮点数
def string_float(text):
    string_float = 1.0
    if text.find('^') != -1 or text.find('**') != -1:
        if text.find('^') != -1:
            str_list = text.split("^", 1)
            string_float = math.pow(float(str_list[0]), float(str_list[1]))
        else:
            str_list = text.split("**", 1)
            string_float = math.pow(float(str_list[0]), float(str_list[1]))
        return string_float
    else:
        try:
            string_float = float(text)
            return string_float
        except:
            return string_float


# 训练模式
def train_model(parameters_data):
    global save_path  # 修改全局变量前声明
    while True:
        str_in = input('是否继续：')
        if str_in == '0' or str_in == 'n' or str_in == 'N' or str_in == 'no' or str_in == 'NO':
            break
        elif str_in == '1' or str_in == 'y' or str_in == 'Y' or str_in == 'yes' or str_in == 'YES':
            parameters_data = NN.train_final(parameters_data)

            save_in = input('是否保存模型：')
            if save_in == '1' or save_in == 'y' or save_in == 'Y' or save_in == 'yes' or save_in == 'YES':
                if save_path == "":
                    save_path = input('请输入保存路径：')
                    print(NN.parameters_data_save(parameters_data, save_path))  # 保存模型
                else:
                    save_path_change_in = input('是否更换路径：')
                    if  save_path_change_in == '1' or  save_path_change_in == 'y' or  save_path_change_in == 'Y' or  save_path_change_in == 'yes' or  save_path_change_in == 'YES':
                        save_path = input('请输入保存路径：')
                        print(NN.parameters_data_save(parameters_data, save_path))  # 保存模型
                    else:
                        print(NN.parameters_data_save(parameters_data, save_path))  # 保存模型
        else:
            break


if __name__ == '__main__':

    while True:
        print("请选择模式：\n模式1：新建模型并训练模式\n模式2：获取最好learn_rate\n模式3：读入模型并训练模式\n模式4：退出\n")
        str_pattern = input('模式：')
        if str_pattern == "1":
            # 新建模型
            learn_rate = string_float(input('请输入learn_rate（学习率）：'))  # 学习率
            epoch_num = int(input('请输入epoch_num（Epoch次数）：'))  # Epoch（时期）次数
            batch_size = int(input('请输入batch_size（分组大小）：'))  # 分组

            parameters = NN.init_parameters()  # 初始化
            current_epoch = 0  # 训练了多少个epoch
            train_loss_list = []
            valid_loss_list = []
            train_accu_list = []
            valid_accu_list = []

            parameters_data = NN.parameters_data_encoded(parameters, learn_rate, epoch_num, batch_size, current_epoch,
                                                         train_loss_list, valid_loss_list, train_accu_list,
                                                         valid_accu_list)

            # print(NN.valid_accuracy(parameters_data[0]['parameters']))  # 训练前正确率

            # 训练模式
            train_model(parameters_data)

        elif str_pattern == "2":
            while True:
                str_in = str_pattern = input('是否继续：')
                if str_in == '0' or str_in == 'n' or str_in == 'N' or str_in == 'no' or str_in == 'NO':
                    break
                elif str_in == '1' or str_in == 'y' or str_in == 'Y' or str_in == 'yes' or str_in == 'YES':
                    batch_size = int(input('请输入batch_size（分组大小）：'))

                    parameters = NN.init_parameters()  # 初始化

                    rand_batch = np.random.randint(NN.train_num // batch_size)
                    grad_lr = NN.train_batch(rand_batch, parameters, batch_size)

                    lr_list = []
                    lower = string_float(input('请输入lower（下限）：'))
                    upper = string_float(input('请输入upper（上限）：'))
                    step = int(input('请输入step（间隔）：'))
                    NN.get_loss(lr_list, lower, upper, step, grad_lr)
                else:
                    break
        elif str_pattern == "3":
            open_path = input('请输入模型路径：')
            try:
                parameters_data = NN.parameters_data_open(open_path)  # 读入模型
                print("读入成功\n正确率："+str(NN.valid_accuracy(parameters_data[0]['parameters']))+"\n进入训练")  # 训练前正确率

                train_configure_in=input('是否使用模型原配置：')
                if train_configure_in == '1' or train_configure_in == 'y' or train_configure_in == 'Y' or train_configure_in == 'yes' or train_configure_in == 'YES':
                    # 训练模式
                    train_model(parameters_data)
                else:
                    # 重新输入配置
                    learn_rate = string_float(input('请输入learn_rate（学习率）：'))  # 学习率
                    epoch_num = int(input('请输入epoch_num（Epoch次数）：'))  # Epoch（时期）次数
                    batch_size = int(input('请输入batch_size（分组大小）：'))  # 分组

                    parameters = parameters_data[0]['parameters']  # 初始化
                    current_epoch = parameters_data[4]['current_epoch']  # 训练了多少个epoch
                    train_loss_list = parameters_data[5]['train_loss_list']
                    valid_loss_list = parameters_data[6]['valid_loss_list']
                    train_accu_list = parameters_data[7]['train_accu_list']
                    valid_accu_list = parameters_data[8]['valid_accu_list']

                    parameters_data = NN.parameters_data_encoded(parameters, learn_rate, epoch_num, batch_size,
                                                                 current_epoch,
                                                                 train_loss_list, valid_loss_list, train_accu_list,
                                                                 valid_accu_list)

                    # 训练模式
                    train_model(parameters_data)
            except:
                print("读入失败")
                break
        elif str_pattern == "4":
            break
        else:
            break