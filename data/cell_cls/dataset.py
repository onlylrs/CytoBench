import os
import torch
from torch.utils.data import Dataset
import PIL.Image as Image
import json
import numpy as np

# Dataset-specific label file paths configuration
DATASET_LABEL_PATHS = {
    'All-IDB': {
        'train': '/jhcnas4/jh/cytology/CYTO_task/All-IDB/train.txt',
        'val': '/jhcnas4/jh/cytology/CYTO_task/All-IDB/val.txt',
        'test': '/jhcnas4/jh/cytology/CYTO_task/All-IDB/test.txt'
    },
    'AML': {
        'train': '/jhcnas4/jh/cytology/CYTO_task/AML/train.txt',
        'val': '/jhcnas4/jh/cytology/CYTO_task/AML/val.txt',
        'test': '/jhcnas4/jh/cytology/CYTO_task/AML/test.txt'
    },
    'Ascites2020': {
        'train': '/jhcnas4/jh/cytology/CYTO_task/Ascites2020/cls/train.txt',  
        'val': '/jhcnas4/jh/cytology/CYTO_task/Ascites2020/cls/val.txt',    
        'test': '/jhcnas4/jh/cytology/CYTO_task/Ascites2020/cls/test.txt'    
    },
    'Barcelona': {
        'train': '/jhcnas4/jh/cytology/CYTO_task/Barcelona/train.txt',  
        'val': '/jhcnas4/jh/cytology/CYTO_task/Barcelona/val.txt',
        'test': '/jhcnas4/jh/cytology/CYTO_task/Barcelona/test.txt' 
    },
    'BCFC': {
        'train': '/jhcnas4/jh/cytology/CYTO_task/BCFC/train.txt',  
        'val': '/jhcnas4/jh/cytology/CYTO_task/BCFC/val.txt',
        'test': '/jhcnas4/jh/cytology/CYTO_task/BCFC/test.txt' 
    },
    'BCI': {
        'train': '/jhcnas4/jh/cytology/CYTO_task/BCI/cls/train.txt',  
        'val': '/jhcnas4/jh/cytology/CYTO_task/BCI/cls/val.txt',    
        'test': '/jhcnas4/jh/cytology/CYTO_task/BCI/cls/test.txt'    
    },
    'BMC': {
        'train': '/jhcnas4/jh/cytology/CYTO_task/BMC/train.txt',  
        'val': '/jhcnas4/jh/cytology/CYTO_task/BMC/val.txt',
        'test': '/jhcnas4/jh/cytology/CYTO_task/BMC/test.txt' 
    },
    'BMT': {
        'train': '/jhcnas4/jh/cytology/CYTO_task/BMT/train.txt',  
        'val': '/jhcnas4/jh/cytology/CYTO_task/BMT/val.txt',
        'test': '/jhcnas4/jh/cytology/CYTO_task/BMT/test.txt' 
    },
    'Breast2023': {
        'train': '/jhcnas4/jh/cytology/CYTO_task/Breast2023/train.txt',  
        'val': '/jhcnas4/jh/cytology/CYTO_task/Breast2023/val.txt',
        'test': '/jhcnas4/jh/cytology/CYTO_task/Breast2023/test.txt' 
    },
    'C_NMC_2019': {
        'train': '/jhcnas4/jh/cytology/CYTO_task/C_NMC_2019/train.txt',  
        'val': '/jhcnas4/jh/cytology/CYTO_task/C_NMC_2019/val.txt',
        'test': '/jhcnas4/jh/cytology/CYTO_task/C_NMC_2019/test.txt'  
    },
    'CCS_Cell_Cls': {
        'train': '/jhcnas4/jh/cytology/CYTO_task/CCS_Cell_Cls/train.txt',  
        'val': '/jhcnas4/jh/cytology/CYTO_task/CCS_Cell_Cls/val.txt',
        'test': '/jhcnas4/jh/cytology/CYTO_task/CCS_Cell_Cls/test.txt'  
    },
    'CERVIX93': {
        'train': '/jhcnas4/jh/cytology/CYTO_task/CERVIX93/train.txt',  
        'val': '/jhcnas4/jh/cytology/CYTO_task/CERVIX93/test.txt',
        'test': '/jhcnas4/jh/cytology/CYTO_task/CERVIX93/test.txt'  
    },
    'CSF2022': {
        'train': '/jhcnas4/jh/cytology/CYTO_task/CSF2022/train.txt',  
        'val': '/jhcnas4/jh/cytology/CYTO_task/CSF2022/val.txt',
        'test': '/jhcnas4/jh/cytology/CYTO_task/CSF2022/test.txt'  
    },
    'FNAC2019': {
        'train': '/jhcnas4/jh/cytology/CYTO_task/FNAC2019/train.txt',  
        'val': '/jhcnas4/jh/cytology/CYTO_task/FNAC2019/val.txt',
        'test': '/jhcnas4/jh/cytology/CYTO_task/FNAC2019/test.txt'  
    },
    'Herlev': {
        'train': '/jhcnas4/jh/cytology/CYTO_task/Herlev/cls/train.txt',  
        'val': '/jhcnas4/jh/cytology/CYTO_task/Herlev/cls/val.txt',
        'test': '/jhcnas4/jh/cytology/CYTO_task/Herlev/cls/test.txt'  
    },
    'HiCervix': {
        'train': '/jhcnas4/jh/cytology/CYTO_task/HiCervix/train.txt',  
        'val': '/jhcnas4/jh/cytology/CYTO_task/HiCervix/val.txt',
        'test': '/jhcnas4/jh/cytology/CYTO_task/HiCervix/test.txt'  
    },
    'JinWooChoi': {
        'train': '/jhcnas4/jh/cytology/CYTO_task/JinWooChoi/train.txt',  
        'val': '/jhcnas4/jh/cytology/CYTO_task/JinWooChoi/val.txt',
        'test': '/jhcnas4/jh/cytology/CYTO_task/JinWooChoi/test.txt'  
    },
    'LDCC': {
        'train': '/jhcnas4/jh/cytology/CYTO_task/LDCC/train.txt',  
        'val': '/jhcnas4/jh/cytology/CYTO_task/LDCC/val.txt',
        'test': '/jhcnas4/jh/cytology/CYTO_task/LDCC/test.txt'  
    },
    'MendeleyLBC': {
        'train': '/jhcnas4/jh/cytology/CYTO_task/MendeleyLBC/train.txt',  
        'val': '/jhcnas4/jh/cytology/CYTO_task/MendeleyLBC/val.txt',
        'test': '/jhcnas4/jh/cytology/CYTO_task/MendeleyLBC/test.txt'  
    },
    'PS3C': {
        'train': '/jhcnas4/jh/cytology/CYTO_task/PS3C/train.txt',  
        'val': '/jhcnas4/jh/cytology/CYTO_task/PS3C/val.txt',
        'test': '/jhcnas4/jh/cytology/CYTO_task/PS3C/test.txt'  
    },
    'Raabin_WBC': {
        'train': '/jhcnas4/jh/cytology/CYTO_task/Raabin_WBC/Train.txt',  
        'val': '/jhcnas4/jh/cytology/CYTO_task/Raabin_WBC/TestA.txt',
        'test': '/jhcnas4/jh/cytology/CYTO_task/Raabin_WBC/TestB.txt'  
    },
    'RepoMedUNM': {
        'train': '/jhcnas4/jh/cytology/CYTO_task/RepoMedUNM/train.txt',  
        'val': '/jhcnas4/jh/cytology/CYTO_task/RepoMedUNM/val.txt',
        'test': '/jhcnas4/jh/cytology/CYTO_task/RepoMedUNM/test.txt'  
    },
    'SIPaKMeD': {
        'train': '/jhcnas4/jh/cytology/CYTO_task/SIPaKMeD/train.txt',  
        'val': '/jhcnas4/jh/cytology/CYTO_task/SIPaKMeD/val.txt',
        'test': '/jhcnas4/jh/cytology/CYTO_task/SIPaKMeD/test.txt'  
    },
    'Thyroid2024': {
        'train': '/jhcnas4/jh/cytology/CYTO_task/Thyroid2024/train.txt',  
        'val': '/jhcnas4/jh/cytology/CYTO_task/Thyroid2024/val.txt',
        'test': '/jhcnas4/jh/cytology/CYTO_task/Thyroid2024/test.txt'  
    },
    'UFSC_OCPap': {
        'train': '/jhcnas4/jh/cytology/CYTO_task/UFSC_OCPap/cls/train.txt',  
        'val': '/jhcnas4/jh/cytology/CYTO_task/UFSC_OCPap/cls/val.txt',
        'test': '/jhcnas4/jh/cytology/CYTO_task/UFSC_OCPap/cls/test.txt'  
    }
}

class CellClsDataset(Dataset):
    """
    Dataset for cell classification

    Args:
        dataset_name (str): Name of the dataset for label path lookup
        preprocess (callable): Preprocessing function for images
        split (str): Dataset split ('train', 'val', or 'test')
        root (str, optional): Root directory of the dataset (fallback if dataset_name not in DATASET_LABEL_PATHS)
    """
    def __init__(self, dataset_name, preprocess, split='train', root=None):
        self.dataset_name = dataset_name
        self.preprocess = preprocess
        self.split = split
        self.root = root

        # Set task type
        self.task_type = "cls"

        # Load dataset metadata
        self.metadata_path = self._get_metadata_path()
        if self.metadata_path and os.path.exists(self.metadata_path):
            self.metadata = self._load_metadata_from_file()
        else:
            # If metadata file doesn't exist, scan the directory structure
            self.metadata = self._create_metadata()

        # Get class labels
        self.label_dict = self._get_label_dict()

    def _get_metadata_path(self):
        """Get the metadata file path based on dataset configuration"""
        # First check if dataset has custom label paths configured
        if self.dataset_name and self.dataset_name in DATASET_LABEL_PATHS:
            custom_path = DATASET_LABEL_PATHS[self.dataset_name].get(self.split, '')
            if custom_path and custom_path.strip():  # Check if path is not empty
                return custom_path

        # Fall back to default path structure if root is provided
        if self.root:
            return os.path.join(self.root, f"{self.task_type}/{self.split}.txt")

        # If no root provided and dataset not in DATASET_LABEL_PATHS, raise error
        raise ValueError(f"Dataset '{self.dataset_name}' not found in DATASET_LABEL_PATHS and no root directory provided")
        
    def _load_metadata_from_file(self):
        """Load metadata from split file"""
        metadata = []
        with open(self.metadata_path, 'r') as f:
            lines = f.readlines()

        # Extract root directory from the metadata file path if not provided
        if not self.root and self.dataset_name in DATASET_LABEL_PATHS:
            # Get the directory containing the metadata file
            metadata_dir = os.path.dirname(self.metadata_path)
            self.root = metadata_dir

        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 2:
                # Handle both absolute and relative paths
                image_rel_path = parts[0]
                if os.path.isabs(image_rel_path):
                    image_path = image_rel_path
                else:
                    image_path = os.path.join(self.root, image_rel_path)

                label = parts[-1]
                metadata.append({
                    'image_path': image_path,
                    'label': label
                })
            else:
                print(f"Warning: Invalid line format in {self.metadata_path}: {line.strip()}")

        return metadata
        
    def _create_metadata(self):
        """Create metadata by scanning directory structure"""
        metadata = []

        if not self.root:
            raise ValueError(f"Cannot create metadata for dataset '{self.dataset_name}': no root directory available and metadata file not found")

        # Get class directories
        class_dirs = [d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))]

        for class_dir in class_dirs:
            class_path = os.path.join(self.root, class_dir)

            # Check if this directory has split subdirectories
            split_path = os.path.join(class_path, self.split)
            if os.path.exists(split_path):
                # If split subdirectory exists, use it
                image_dir = split_path
            else:
                # Otherwise use the class directory directly
                image_dir = class_path

            # Get all image files
            image_files = [f for f in os.listdir(image_dir)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]

            for img_file in image_files:
                metadata.append({
                    'image_path': os.path.join(image_dir, img_file),
                    'label': class_dir
                })

        return metadata
    
    def _get_label_dict(self):
        """Create a dictionary mapping class names to indices"""
        labels = sorted(list(set([item['label'] for item in self.metadata])))
        return {label: idx for idx, label in enumerate(labels)}
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        item = self.metadata[idx]
        
        # Load and preprocess image
        image_path = item['image_path']
        image = Image.open(image_path).convert('RGB')
        image = self.preprocess(image)
        
        # Get label index
        label = self.label_dict[item['label']]
        
        return image, label

def build_cell_cls_dataloaders(config):
    """
    Build dataloaders for cell classification tasks
    
    Args:
        config (dict): Configuration dictionary containing dataset settings
        
    Returns:
        dict: Dictionary of dataloaders for each dataset
    """
    from torch.utils.data import DataLoader
    from backbone.build_model import build_backbone
    
    # Build backbone to get preprocessing function
    _, preprocess = build_backbone(config)
    
    # Get dataset names from config or use default list
    dataset_names = config.get('data', {}).get('datasets', 
                              ['Herlev', 'HiCervix', 'JinWooChoi', 'FNAC2019', 
                               'LDCC', 'Sipakmed', 'Barcelona', 'BCI', 'BCFCI', 'BMCC'])
    
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    
    dataloaders = {}
    
    for dataset_name in dataset_names:
        # Get root directory if provided in config (fallback)
        dataset_root = None
        if 'root' in config.get('data', {}):
            dataset_root = os.path.join(config['data']['root'], dataset_name)

        # Create datasets for each split
        train_dataset = CellClsDataset(
            dataset_name=dataset_name,
            preprocess=preprocess,
            split='train',
            root=dataset_root
        )

        val_dataset = CellClsDataset(
            dataset_name=dataset_name,
            preprocess=preprocess,
            split='val',
            root=dataset_root
        )

        test_dataset = CellClsDataset(
            dataset_name=dataset_name,
            preprocess=preprocess,
            split='test',
            root=dataset_root
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['common']['num_workers'],
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['evaluation']['batch_size'],
            shuffle=False,
            num_workers=config['common']['num_workers'],
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['evaluation']['batch_size'],
            shuffle=False,
            num_workers=config['common']['num_workers'],
            pin_memory=True
        )
        
        # Store dataloaders for this dataset
        dataloaders[dataset_name] = {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader,
            'num_classes': len(train_dataset.label_dict)
        }
    
    return dataloaders
