from collect_dataset import run_collect

# Define
run_time = 60  # second
action_select = 3 # 0=dribble, 1=shoot, 2=pass, 3=stand
path_dataset = 'F:\\Master Project\\Dataset\\BasketBall-RGB\\' # folder path
show_FPS = False

action_list = ['dribble','shoot','pass','stand']
action = action_list[action_select]
path_save = path_dataset +'\\'+action+'\\'+action

run_collect(path_save, run_time, show_FPS)
print("finish main")
