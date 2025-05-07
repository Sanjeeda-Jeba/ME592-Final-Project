# ME5920 Final Project

**Deadline**: 7 May

## Project Title: 
**Robust Autonomous Driving System Using Deep Learning and Generative AI**

Datasets: https://github.com/lhyfst/awesome-autonomous-driving-datasets

---

## Person 1: Jamie Preston  
**Role**: Data Collection + Baseline CNN Model (Behavior Cloning)

### Tasks:
- Set up and use simulator (CARLA or Udacity simulator).
- Record clean daytime driving data (images + steering, brake, throttle).
- Build baseline CNN model:
  - Custom CNN or pre-trained (e.g., ResNet18) fine-tuned for regression.
  - Output 3 values: steering, brake, throttle.
- Train and validate the baseline model on clean data.

### Deliverables:
- Dataset (images + control labels)
- Baseline model training script (`baseline_behavior_cloning.py`)
- Training curves: loss, MAE plots
- Saved model checkpoints

---

## Person 2: Angona Biswas  
**Role**: Generative AI Part (Data Augmentation)

### Tasks:
- Implement image augmentation to simulate night, rain, fog conditions.
  - Option 1: Use image transformation libraries (Albumentations, torchvision)
  - Option 2: Use lightweight GAN models to generate environmental effects
- Augment the clean dataset to create challenging conditions dataset.
- Retrain the behavior cloning model on augmented + clean data (fine-tuning).
- Track model improvement after augmentation.

### Deliverables:
- Augmented dataset
- Data augmentation scripts (`augment_images.py`)
- Fine-tuned model training script (`behavior_cloning_augmented.py`)
- Plots: loss, MAE comparison (clean-only vs augmented training)

---

## Person 3: [Your Name]  
**Role**: Improvement Trial, Evaluation, Report Writing, and Presentation

### Tasks:
- Design and perform evaluations:
  - Compare performance on clean vs foggy/night/rainy unseen environments.
  - Plot actual vs predicted steering/throttle/brake.
  - Metrics: MAE, RMSE, or other regression evaluation metrics.
- Write the final report in research paper format:
  - Sections: Abstract, Introduction, Methodology, Experiments, Results, Discussion, Conclusion.
- Prepare presentation slides (10–12 slides).
- Maintain GitHub repo:
  - Organize all codes, datasets, and models.
  - Include disclosure and citations for any external tools/models (e.g., CARLA, pre-trained networks).

### Deliverables:
- Evaluation report (`evaluation_metrics.ipynb`)
- Full Project Report (PDF/Word in paper format)
- Final Presentation Slides (Google Slides / PowerPoint)
- Organized GitHub Repo

---

### Note:
- For report and presentation:
  - Each team member contributes content related to their work.
  - The person responsible for a section should ideally write and present that section in class.
