import torch
import torchvision
from torch.utils.data import Dataset
import mediapipe as mp
from tqdm import tqdm
import numpy as np
import os
import cv2

## utils
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh



class DISFA(Dataset) :
    def __init__(self, root, transform=None, target_transform=None) :
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        self.target_directory = os.path.abspath(os.path.join(os.path.dirname(self.root), 'pre_processed', 'disfa'))

        if not os.path.exists(self.target_directory) :
            print("Creating target directory...")
            os.makedirs(self.target_directory)
        else :
            print("Target directory already exists...")
        self.face_mesh = mp_face_mesh.FaceMesh(max_num_faces = 1,
                                   refine_landmarks=True, 
                                   min_detection_confidence=0.5, 
                                   min_tracking_confidence=0.5)

        self.prepare_data()

    # def _extract_3dmesh(self) :

    # def _crop_image(self) :


    def extract_frame(self, video, person, mesh3d=False, crop=False) :
        # extract frame from video
        right = True if "Right" in video else False
        # extract 3d mesh from video
        # crop image
        target_image_path = os.path.join(self.target_directory, "images", "right_frame" if right else "left_frame", person )
        if not os.path.exists(target_image_path) :
            os.makedirs(target_image_path)

        target_mesh3d_path = os.path.join(self.target_directory, "mesh", "right_mesh" if right else "left_mesh", person)
        if not os.path.exists(target_mesh3d_path) :
            os.makedirs(target_mesh3d_path)

        if crop : 
            target_crop_path = os.path.join(self.target_directory, "images", "crop", person)
            if not os.path.exists(target_crop_path) :
                os.makedirs(target_crop_path)

        vidcap = cv2.VideoCapture(video)
        count = 0
        video_length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1        
        with tqdm(total=video_length) as pbar:
            while vidcap.isOpened():
                success, image = vidcap.read()
                if not success:
                    break
                cv2.imwrite( os.path.join(target_image_path, f"{person}_{count:04d}.jpg"), image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(image)
                face = results.multi_face_landmarks
                if face is not None : 
                    landmarks = np.array([ (ld.x, ld.y, ld.z) for ld in face[0].landmark])
                    if mesh3d :
                        np.save(os.path.join(target_mesh3d_path, f"{person}_{count:04d}.npy"), landmarks)
                        if crop :
                            H, W = image.shape[:2]
                            _min_x, _min_y, _ =  np.min(landmarks, axis = 0)
                            _max_x, _max_y, _ =  np.max(landmarks, axis = 0)
                            height = _max_y - _min_y
                            width  = _max_x - _min_x
                            # +30% of the face width and height
                            x_min, y_min , x_max , y_max = _min_x - 0.20*width,  _min_y - 0.2*height,  _max_x + 0.2*width, _max_y + 0.2*height
                            x_min, y_min , x_max , y_max = np.int0(x_min*W), np.int0(y_min*H), np.int0(x_max*W), np.int0(y_max*H)
                            # crop image
                            image = image[y_min:y_max, x_min:x_max]
                            # resize image
                            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                            image = cv2.resize(image, dsize=(256, 256), interpolation = cv2.INTER_AREA)
                            # save image
                            cv2.imwrite(os.path.join(self.target_directory, "images", "crop", person, f"{person}_{count:04d}.jpg") , image)
                count += 1
                pbar.update(1)


    def extract_action_units(self, au_dir, person) :
        target_au_directory = os.path.join(self.target_directory, "action_units", person)
        if not os.path.exists(target_au_directory) :
            os.makedirs(target_au_directory)
        files = os.listdir(au_dir)
        aus = sorted([ x[len('SN001_'):-len(".txt")] for x in files], key = lambda u : int(u[2:]))

        all_au = []
        for au in aus :
            au_path = os.path.join(au_dir, f"{person}_{au}.txt")
            with open(au_path, 'r') as f :
                au_content = f.readlines()
            f.close()
            au_content = [ int(line.strip().split(",")[1]) for line in au_content]
            all_au.append(au_content)
        
        # au_per_frame = zip(*all_au)
        au_per_frame = [ dict(zip(aus, frame)) for frame in zip(*all_au) ]

        for i, file_ in enumerate(au_per_frame) :
            np.save(os.path.join(target_au_directory, f"{person}_frame{i:04d}.npy"), file_)
    





    def prepare_data(self) :
        action_units = os.path.join(self.root, 'ActionUnitsLabels')
        images_right = os.path.join(self.root, 'Right_Video')
        images_left = os.path.join(self.root, 'Left_Video')

        ids = os.listdir(action_units)
        for id_ in tqdm(ids) :
            # extract right frame
            self.extract_frame(os.path.join(images_right, f"RightVideo{id_}_comp.avi"), 
                               person = id_, mesh3d=True, crop=True)
            # # extract left frame
            self.extract_frame(os.path.join(images_left, f"LeftVideo{id_}_comp.avi"), 
                               person = id_, mesh3d=True, crop=False)
            # extract Action Units
            self.extract_action_units(os.path.join(action_units, id_), id_)




    # def __getitem__(self, index) :
    #     if self.train :
    #         img, target = self.train_data[index], self.train_labels[index]
    #     if self.test :
    #         img, target = self.test_data[index], self.test_labels[index]

    #     if self.transform is not None :
    #         img = self.transform(img)
    #     if self.target_transform is not None :
    #         target = self.target_transform(target)

    #     return img, target

    # def __len__(self) :
    #     if self.train :
    #         return len(self.train_data)
    #     if self.test :
    #         return len(self.test_data)

if __name__ == '__main__' :
    dataset = DISFA(root='./data/DISFA', transform=None, target_transform=None)
    print(dataset.root, dataset.target_directory)