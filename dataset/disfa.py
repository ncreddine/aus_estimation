import torch
import torchvision
from torch.utils.data import Dataset
import mediapipe as mp
from tqdm import tqdm
import numpy as np
import skimage
import os
import cv2
import json

class Disfa(Dataset) :
    def __init__(self, root, transform=None, target_transform=None, folds = [2,1], seed = 42, current_fold = 0) :
        # Original dataset
        self.root = root
        # Apply transformation on the train and the test set
        self.transform = transform
        self.target_transform = target_transform

        if not self.if_preprocessed() :
            self.prepare_data()

        self.train_persons, self.test_persons = self.split_persons(folds, seed, current_fold = current_fold)

        self.train_images, self.train_mesh3d_right, self.train_mesh3d_left, self.train_labels = self.load_data(self.train_persons)
        self.test_images, self.test_mesh3d_right, self.test_mesh3d_left, self.test_labels = self.load_data(self.test_persons)

        # self.weights = self.compute_weights(self.train_labels, self.test_labels)

        # self.train_images, self.train_mesh3d_right, self.train_mesh3d_left, self.train_labels = self.train_images[:1000], self.train_mesh3d_right[:1000], self.train_mesh3d_left[:1000], self.train_labels[:1000]
        # self.test_images, self.test_mesh3d_right, self.test_mesh3d_left, self.test_labels = self.test_images[:1000], self.test_mesh3d_right[:1000], self.test_mesh3d_left[:1000], self.test_labels[:1000]


        self.train_dataset = AuDataset(transform=self.transform, target_transform=self.target_transform, images=self.train_images, meshes=self.train_mesh3d_right, labels=self.train_labels)
        self.test_dataset = AuDataset(transform=self.transform, target_transform=self.target_transform, images=self.test_images, meshes=self.test_mesh3d_right, labels=self.test_labels)



    def compute_weights(self, labels_train, labels_test) :
        labels = list([*labels_train, *labels_test])
        labels_ = []
        for labels_file in tqdm(labels, desc="Compute Weight Matrix") :
            content = list(np.load(labels_file, allow_pickle=True).flat[0].values())
            labels_.append(content)
        
        labels_ = np.array(labels_).T

        histogram = []
        for lab in labels_ :
            hist = np.histogram(lab, bins=np.arange(7))[0]
            histogram.append(hist)
        histogram = np.array(histogram)

        return torch.tensor(histogram)




    def split_persons(self, folds, seed, current_fold = 0): 
        persons = self.persons
        persons_per_fold = len(persons) // sum(folds)

        # shuffle persons
        np.random.seed(seed)
        np.random.shuffle(persons)

        num_test_persons = persons_per_fold * folds[1]

        test_set = persons[current_fold * persons_per_fold : current_fold * persons_per_fold + num_test_persons]
        train_set = [ p for p in persons if p not in test_set]
        
        return train_set, test_set


    def load_data(self, persons) :
        images = []
        meshes_r = []
        meshes_l = []
        labels = []
        for person in persons :
            person_images = os.path.join(self.target_directory, "images", "crop", person)
            person_meshes_r = os.path.join(self.target_directory, "facemesh", "right", person)
            person_meshes_l = os.path.join(self.target_directory, "facemesh", "left", person)
            person_labels = os.path.join(self.target_directory, "action_units", person)
            images += [ os.path.join(person_images, x) for x in sorted(os.listdir(person_images))]
            meshes_r += [ os.path.join(person_meshes_r, x) for x in sorted(os.listdir(person_meshes_r))]
            meshes_l += [ os.path.join(person_meshes_l, x) for x in sorted(os.listdir(person_meshes_l))]
            labels += [ os.path.join(person_labels, x) for x in sorted(os.listdir(person_labels))]
        return images, meshes_r, meshes_l, labels    

    @property
    def persons(self) :
        return list(sorted(os.listdir(os.path.join(self.root, 'ActionUnits_Labels'))))

    def if_preprocessed(self) :
        self.target_directory = os.path.abspath(os.path.join(os.path.dirname(self.root), 'pre_processed', 'disfa'))
        self.status_file = os.path.join(self.target_directory, "status.txt")
        if not os.path.exists(self.target_directory) :
            print("Creating target directory...")
            os.makedirs(self.target_directory)
            return False
        else :
            print("Target directory already exists...")
            return self.if_all_files_preprocessed()

    def if_all_files_preprocessed(self) :
        print("Checking if all files are preprocessed...")
        if not os.path.exists(self.status_file) :
            return False
        else :
            with open(self.status_file, 'r') as f :
                status = f.readlines()
            return len(status) == len(self.persons)
    
    def prepare_data(self) :
        # Path to action units, images and 3d mesh
        action_units = os.path.join(self.root, 'ActionUnits_Labels')
        # images = os.path.join() # here we only consider right video

        ids = sorted(os.listdir(action_units))

        with tqdm(total=len(ids), bar_format="{desc:<15}{percentage:3.0f}%|{bar:50}{r_bar}", leave = False)  as pbar_ :
            for id_ in ids :
                pbar_.set_description(f"Person {id_}")

                # Save action units
                nb_annotated_frames = self.read_save_action_units(os.path.join(action_units, id_), id_)

                # Save whole frame, 3d mesh and cropped image
                # self.read_save_frame(os.path.join(self.root, 'Video_RightCamera', f"RightVideo{id_}_comp.avi"),
                #                      person = id_, nb_annotated_frames = nb_annotated_frames, save_frame = True, save_mesh3d = True, save_crop=True)
                # Left
                self.read_save_frame(os.path.join(self.root, 'Video_LeftCamera', f"LeftVideo{id_}_comp.avi"),
                                     person = id_, nb_annotated_frames = nb_annotated_frames, save_frame = False, save_mesh3d = True, save_crop=False)
                # Update status
                with open(self.status_file, 'a') as f :
                   f.write(f"{id_}\n")
                f.close() 
                pbar_.update(1)


    @staticmethod
    def crop_image(landmarks, image) :
        margin = 0.3
        shape = image.shape[:2]
        # bounding box of the face
        [min_x, min_y], [max_x, max_y] = np.min(landmarks[:,:2], axis = 0), np.max(landmarks[:,:2], axis = 0)
        height = max_y - min_y ; width  = max_x - min_x
        # +30% of the face width and height
        x_min, y_min , x_max , y_max = min_x - margin * width,  min_y - margin * height,  max_x + margin * width, max_y + margin * height
        x_min , y_min = max(0, int(x_min * shape[1])) , max(0, int(y_min * shape[0])) 
        x_max , y_max = min(shape[1], int(x_max * shape[1])), min(shape[1], int(y_max * shape[0]))
        return  image[y_min:y_max, x_min:x_max]


    def read_save_frame(self, video, person, nb_annotated_frames, save_frame = True, save_mesh3d = False, save_crop = False) :
        """Extract frame from video and save them in the target directory"""
        # Frames Directory
        target_image_path = os.path.join(self.target_directory, "images", "frame" , person )
        if not os.path.exists(target_image_path) :
            os.makedirs(target_image_path)

        # Facemesh Directory
        target_mesh3d_path = os.path.join(self.target_directory, "facemesh", "right" if "right" in video.lower() else "left", person)
        if not os.path.exists(target_mesh3d_path) :
            os.makedirs(target_mesh3d_path)

        # Crop images directory
        if save_crop : 
            target_crop_path = os.path.join(self.target_directory, "images", "crop", person)
            if not os.path.exists(target_crop_path) :
                os.makedirs(target_crop_path)

        # Read video 
        vidcap = cv2.VideoCapture(video)
        video_length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1        
        with tqdm(total=video_length, bar_format="{desc:<15}{percentage:3.0f}%|{bar:50}{r_bar}", leave = False)  as pbar :
            with  mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, refine_landmarks=True, min_detection_confidence=0.8, min_tracking_confidence=0.8) as face_mesh :
                count = 0
                while vidcap.isOpened() and count < nb_annotated_frames :
                    # read frame
                    success, image = vidcap.read()
                    if not success:
                        break
                    # save frame
                    if save_frame : target_frame = os.path.join(target_image_path, f"{person}_frame_{count:04d}.jpg")
                    if save_mesh3d : target_mesh = os.path.join(target_mesh3d_path, f"{person}_mesh_{count:04d}.npy")
                    if save_crop : target_crop = os.path.join(target_crop_path, f"{person}_crop_{count:04d}.jpg")
                    if save_frame and save_mesh3d and save_crop :
                        if os.path.exists(target_frame) and os.path.exists(target_crop) and os.path.exists(target_mesh) :
                            count += 1
                            pbar.update(1)
                            continue
                        if save_frame : cv2.imwrite(target_frame, image)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(image)
                    face = results.multi_face_landmarks
                    if face is not None : 
                        landmarks = np.array([ (ld.x, ld.y, ld.z) for ld in face[0].landmark])
                        if save_mesh3d :
                            np.save(target_mesh, landmarks)
                            if save_crop :
                                image = self.crop_image(landmarks, image)
                                # resize image
                                image = skimage.transform.resize(image, (256, 256))
                                skimage.io.imsave(target_crop, (image / image.max() * 255).astype(np.uint8))
                    count += 1
                    pbar.update(1)


    def read_save_action_units(self, au_dir, person) :
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

        nb_frames = len(all_au[0])
        
        au_per_frame = [ dict(zip(aus, frame)) for frame in zip(*all_au) ]

        for i, file_ in enumerate(au_per_frame) :
            np.save(os.path.join(target_au_directory, f"{person}_au_{i:04d}.npy"), file_)

        return nb_frames
            


class AuDataset(Dataset) :
    def __init__(self, transform, target_transform, images, meshes, labels ) -> None:
        super(AuDataset, self).__init__()
        self.images = images
        self.meshes = meshes
        self.labels = labels

        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self, index) :
        img, mesh, action_units = self.images[index], self.meshes[index], self.labels[index]

        img = skimage.io.imread(img)
        mesh = torch.from_numpy(np.load(mesh))
        action_units = np.load(action_units, allow_pickle=True).flat[0]

        if self.transform is not None :
            mesh = self.transform(mesh)
        
        if self.target_transform is not None :
            action_units = self.target_transform(action_units)

        return img, mesh, action_units

    def __len__(self) :
        return len(self.meshes)

if __name__ == '__main__' :
    dataset = Disfa(root='./data/DISFA', transform=None, target_transform=None)

    trainset = dataset.train_dataset
    testset = dataset.test_dataset

    print(dataset.weights)

    for img, mesh, au in trainset :
        print(img.shape, mesh.shape, au.keys())
        break