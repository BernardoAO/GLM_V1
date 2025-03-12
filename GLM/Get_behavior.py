{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Get behavior files"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os \n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "working_dir ='/home/ag-schmitz-nwfz/yota/project'\n",
    "# load all the cam files \n",
    "path_2_behavior =  \\\n",
    "    os.path.join(working_dir, 'Facemap/facemap_motion_data')\n",
    "# path to save the file\n",
    "save_path =  \\\n",
    "    os.path.join(working_dir, 'GLM/\"valid_periods.npy\"')\n",
    "\n",
    "# Information\n",
    "camara_sampling_rate = 200 # Hz\n",
    "n_ROIs = 6\n",
    "ROI_names = [\"Wheel\", \"Vibrissae\", \"Nose\", \"Pupil\", \"Mouth\", \"Paw\"]\n",
    "cam_files_name = 'cam0_20'\n",
    "\n",
    "\n",
    "cam_files = [file for file in os.listdir(path_2_behavior) \n",
    "                  if file.startswith(cam_files_name)]\n",
    "cam_files.sort()\n",
    "\n",
    "Behavior = []\n",
    "for file in cam_files:\n",
    "    Behavior.append(np.load(os.path.join(path_2_behavior,file)\n",
    "            ,allow_pickle=True,encoding='latin1').item())\n",
    "\n",
    "current_day = cam_files[0][5:15]\n",
    "start = 0\n",
    "valid_periods = {}\n",
    "p = 0\n",
    "\n",
    "for cam_i, cam in enumerate(Behavior):    \n",
    "\n",
    "    if not cam_files[cam_i][5:15] == current_day:\n",
    "       \n",
    "        #Show plot of all day´s periods\n",
    "        plt.xlabel(\"time [s]\")\n",
    "        plt.tight_layout()\n",
    "        plt.show()\n",
    "\n",
    "        # Prompt the user \n",
    "        periods = input(\"Valid periods (from-to): \")\n",
    "        valid_periods[cam_files[cam_i]] = np.arange(int(periods[0]), int(periods[2]))\n",
    "\n",
    "        #start next plot\n",
    "        current_day = cam_files[cam_i][5:15]\n",
    "        start = 0\n",
    "        plt.figure(figsize=(15,10))\n",
    "\n",
    "    tv = np.arange(start, start + len(cam['motion'][i])) / camara_sampling_rate\n",
    "    \n",
    "    for i in range(1, n_ROIs + 1):\n",
    "        plt.subplot(n_ROIs, 1, i)\n",
    "        plt.plot(tv, cam['motion'][i])\n",
    "        if i < 6:\n",
    "            plt.xticks([])\n",
    "\n",
    "        if p == 0:\n",
    "            plt.title(ROI_names[i - 1])\n",
    "            plt.gca().spines['top'].set_visible(False)\n",
    "            plt.gca().spines['right'].set_visible(False)\n",
    "\n",
    "    p += 1\n",
    "\n",
    "np.save(save_path, valid_periods)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": ".venv",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
