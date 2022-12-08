import fiftyone as fo
import fiftyone.zoo as foz

import os
def load_datasets():
    dataset = foz.load_zoo_dataset(
        "open-images-v6",
        split="validation",
        label_types=["detections", "classifications"],
        classes=["Dog"],
        max_samples=345,
        seed=51,
        shuffle=True,
        dataset_name="dog-subset",
    )
    """
    dog_subset = foz.load_zoo_dataset(
        "open-images-v6",
        split="validation",
        label_types=["detections", "classifications"],
        classes=["Dog"],
        max_samples=30,
        seed=51,
        shuffle=True,
        dataset_name="dog-subset",
    )
    
    _ = dataset.merge_samples(dog_subset)
    """
        
    export_dir = "C:\\Users\\Zac\\Desktop\\datasets\\Dog"
    ##dataset.export(export_dir=export_dir, dataset_type=fo.types.ImageDirectory)
    dataset.export(export_dir=export_dir, dataset_type=fo.types.FiftyOneImageDetectionDataset)
    
    session = fo.launch_app(dataset, desktop=True)
    session.wait()
    
def delete_corrupt():
    from struct import unpack
    from tqdm import tqdm
    import os


    marker_mapping = {
        0xffd8: "Start of Image",
        0xffe0: "Application Default Header",
        0xffdb: "Quantization Table",
        0xffc0: "Start of Frame",
        0xffc4: "Define Huffman Table",
        0xffda: "Start of Scan",
        0xffd9: "End of Image"
    }


    class JPEG:
        def __init__(self, image_file):
            with open(image_file, 'rb') as f:
                self.img_data = f.read()
        
        def decode(self):
            data = self.img_data
            while(True):
                marker, = unpack(">H", data[0:2])
                # print(marker_mapping.get(marker))
                if marker == 0xffd8:
                    data = data[2:]
                elif marker == 0xffd9:
                    return
                elif marker == 0xffda:
                    data = data[-2:]
                else:
                    lenchunk, = unpack(">H", data[2:4])
                    data = data[2+lenchunk:]            
                if len(data)==0:
                    break        


    bads = []

    
    root_img = "src\\data\\pikachu\\data"
    images = os.listdir(root_img)
    
    for img in tqdm(images):
        image = os.path.join(root_img,img)
        image = JPEG(image) 
        try:
            image.decode()   
        except:
            bads.append(img)


    for name in bads:
        os.remove(os.path.join(root_img,name))

def is_image(filename, verbose=False):

    data = open(filename,'rb').read(10)

    # check if file is JPG or JPEG
    if data[:3] == b'\xff\xd8\xff':
        if verbose == True:
             print(filename+" is: JPG/JPEG.")
        return True

    # check if file is PNG
    if data[:8] == b'\x89\x50\x4e\x47\x0d\x0a\x1a\x0a':
        if verbose == True:
             print(filename+" is: PNG.")
        return True

    # check if file is GIF
    if data[:6] in [b'\x47\x49\x46\x38\x37\x61', b'\x47\x49\x46\x38\x39\x61']:
        if verbose == True:
             print(filename+" is: GIF.")
        return True

    return False

if __name__ == '__main__':
    for folder in os.listdir("src\\data"):
     # check if file is actually an image file
        for image in os.listdir(os.path.join("src\\data", folder, "data")):
            filepath = os.path.join("src\\data", folder, "data", image)
            if is_image(filepath, verbose=False) == False:
                # if the file is not valid, remove it
                os.remove(filepath)