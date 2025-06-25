# cell_seg - configs
This directory contains configs for the cell_seg task.

## Configuration Options

### Data Configuration

#### `category_id_offset`
**Purpose**: Maps dataset category IDs to model class IDs to handle different labeling schemes.

**Default**: `1`

**When to use**:
- **Set to `1`**: When your dataset categories start from 0 (e.g., ISBI2014: Cell=0, Nuclei=1)
  - Maps to model classes [1, 2] (background=0, Cell=1, Nuclei=2)
- **Set to `0`**: When your dataset categories already start from 1 
  - Maps to model classes [1, 2] (background=0, existing categories preserved)
- **Set to custom value**: For datasets with non-standard category ID schemes

**Examples**:

```yaml
# ISBI2014 dataset (categories: Cell=0, Nuclei=1)
data:
  dataset: 'ISBI2014'
  category_id_offset: 1  # Maps 0->1, 1->2

# Standard dataset (categories: Cell=1, Nuclei=2)  
data:
  dataset: 'StandardDataset'
  category_id_offset: 0  # Maps 1->1, 2->2

# Custom dataset (categories: Cell=5, Nuclei=10)
data:
  dataset: 'CustomDataset' 
  category_id_offset: -4  # Maps 5->1, 10->6 (may need custom handling)
```

**Important**: Deep learning models expect:
- Class 0 = Background
- Class 1+ = Object classes

Use this option to ensure your dataset's category IDs are correctly mapped to the model's expected class structure.
