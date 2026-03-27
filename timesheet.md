# Plan

* Aiding diagnosis of Multiple Sclerosis with deep learning
* Mark McNaught
* 2764158M
* Dr Chris McCaig

## Winter semester

* **Week 1 (15th Sept - 21st Sept)**
  - Completed project bidding and assigned project.
  - Conducted basic background research on ML and MS.
  - Reviewed available dataset(s) of MRI scans.
  - Planned initial objectives and learning path.

* **Week 2 (22nd Sept - 28th Sept)**
  - Completed some online ML tutorials and coursework.
  - Reviewed foundational ML concepts.
  - Began literature search on MS diagnosis and ML applications.
  - Began online ML course, explored CNNs and transfer learning, reviewed ImageNet models.

* **Week 3 (29th Sept - 5th Oct)**
  - Continued and nearly completed online ML course.
  - Practised ML projects (dogs vs cats classification, lung cancer detection).
  - Continued literature review on MS and ML in medical imaging.
  - Began exploring ResNet models for transfer learning.

* **Week 4 (6th Oct - 12th Oct)**
  - Completed online ML course.
  - Researched ResNet transfer learning implementation.
  - Reviewed research papers in-depth.
  - Identified computational resource limitations as a challenge.

* **Week 5 (13th Oct - 19th Oct)**
  - Created basic CNN model to understand structure and performance.
  - Implemented initial ResNet50 transfer learning model on dataset.
  - Conducted preliminary evaluation: training time, accuracy, overfitting.
  - Issues: GPU resource limitations, small dataset size, class imbalance.

* **Week 6 (20th Oct - 26th Oct)**
  - Implemented ResNet18 model using PyTorch.
  - Analysed and compared performance metrics of CNN and ResNet18 models.
  - Researched additional model architectures and alternative transfer learning approaches.
  - Reviewed additional datasets for potential merging.

* **Week 7 (27th Oct - 2nd Nov)**
  - Modified ResNet18 to incorporate early training cut-off and refined training regime.
  - Plotted and reviewed loss metrics for better model comparison.
  - Researched hybrid model approaches incorporating Vision Transformer features.
  - Began exploring segmentation as a potential project extension.

* **Week 8 (3rd Nov - 9th Nov)**
  - Researched and implemented attention mechanisms (CBAM) into ResNet18.
  - Investigated dataset origin and discovered additional relevant model architectures.
  - Identified suitable additional datasets and discovered RadImageNet.
  - Continued literature review of medical imaging classification approaches.

* **Week 9 (10th Nov - 16th Nov)**
  - Implemented multiple self-attention model variants and compared performances.
  - Reviewed and adapted approaches from existing work on the dataset.
  - Implemented NCA and kNN classifiers on top of existing attention-based models.
  - Began experimenting with TensorBoard for deeper model analysis.

* **Week 10 (17th Nov - 23rd Nov)**
  - Limited progress due to coursework commitments and personal matters.
  - Reviewed and began planning status report.
  - Continued background section research.
  - Discussed project scope and research questions with supervisor.

* **Week 11 [PROJECT WEEK] (24th Nov - 30th Nov)**
  - Limited progress due to continued coursework and personal commitments.
  - Discussed ensemble model approach and dataset split strategies with supervisor.
  - Finalised research question direction around model comparison and evaluation.
  - Established timelines and expectations for second semester.

* **Week 12 [PROJECT WEEK] (1st Dec - 5th Dec)**
  - Prepared and submitted **status report**.
  - Continued planning for ensemble model implementation and dataset experiments.

## Winter break (6th Dec - early Jan)

* Planned codebase refactoring and model improvements.
* Continued background reading and research.
* Addressed supervisor feedback from status report.

## Spring Semester

* **Week 13 (6th Jan - 12th Jan)**
  - Extensive codebase refactoring to improve programming practices and maintainability.
  - Reworked model training workflows and corrected logic issues in training pipeline.
  - Project scope refined towards a comparison of CNN vs Transformer-based architectures.

* **Week 14 (13th Jan - 19th Jan)**
  - Conducted foundational research into Vision Transformers (ViTs) and hybrid architectures.
  - Continued background section write-up.
  - Made incremental progress on codebase refactoring.

* **Week 15 (20th Jan - 26th Jan)**
  - Completed full refactoring of all CNN-related notebooks and model variants.
  - Re-implemented squeeze-and-excitation modules correctly.
  - Continued background section write-up and research.

* **Week 16 (27th Jan - 2nd Feb)**
  - Completed first draft of background section.
  - Continued refactoring NCA and kNN pipelines.
  - Began prototyping ViT models and reviewing relevant literature.
  - Firmed up research questions to cover CNN, ViT, and hybrid model comparisons.

* **Week 17 (3rd Feb - 9th Feb)**
  - Refined and condensed background section based on supervisor feedback.
  - Continued NCA and kNN pipeline refinement.
  - Explored ViT implementations and feasibility within project scope.
  - Confirmed decision to include ViTs to a measured extent alongside CNN work.

* **Week 18 (10th Feb - 16th Feb)**
  - Finalised research questions incorporating ViTs and CNN-ViT hybrid models.
  - Implemented additional model variants to support ablation studies.
  - Implemented train/val/test splits and seeding for reproducibility.
  - Implemented standalone ViT models.
  - Switched referencing style to Harvard and planned full dissertation structure.

* **Week 19 (17th Feb - 23rd Feb)**
  - Wrote first draft of Introduction section.
  - Reworked background section structure and content to align with refined research questions.
  - Implemented stratified k-fold cross-validation.
  - Designed grid search plan for hyperparameter optimisation.

* **Week 20 (24th Feb - 1st Mar)**
  - Designed and implemented grid search for baseline hyperparameter optimisation.
  - Redrafted introduction and background sections.
  - Began designing and implementing hybrid CNN-ViT model.
  - Discussed presentation expectations and dissertation timeline with supervisor.

* **Week 21 (2nd Mar - 8th Mar)**
  - Completed background and introduction sections.
  - Re-implemented hybrid model using a lightweight MHSA-based approach.
  - Collected all experimental results required for dissertation and presentation.
  - Drafted the majority of the dissertation body.
  - Prepared and delivered **final presentation**.

* **Week 22 (9th Mar - 15th Mar)**
  - Completed methodology section and updated research questions for consistency.
  - Planned remaining results, discussion, and conclusion chapters.
  - Refactored codebase and notebooks to align with updated research questions.

* **Week 23 (16th Mar - 22nd Mar)**
  - Focused on completing all remaining dissertation chapters.
  - Reviewed write-up structure, visuals, and overall coherence with supervisor.

* **Week 24 (23rd Mar - 27th Mar)**
  - Completed all remaining sections of the dissertation.
  - Submitted **final dissertation** by March 27th.