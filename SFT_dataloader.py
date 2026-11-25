import os
import glob
import json
import random
from typing import Dict, List, Tuple

random.seed(1234)

from torch.utils.data import Dataset, DataLoader
import torch

def format_list(items):
    if len(items) == 0:
        return ""
    elif len(items) == 1:
        return items[0]
    else:
        return ", ".join(items[:-1]) + " and " + items[-1]

def position_questions(Discs: Dict, Vertbras: Dict) -> List[Tuple[str, str]]:
    qa_pairs = []
    len_disc = len(Discs)
    len_vert = len(Vertbras)

    q = "Which intervertebral discs or vertebrae have lesions or abnormalities?"
    if len_disc > 0 and len_vert > 0:
        discs = list(Discs.keys())
        verts = list(Vertbras.keys())
        a = "Intervertebral disc: " + format_list(discs) + ".\nVertebra: " + format_list(verts) + "."
        qa_pairs.append((q, a))
    elif len_disc > 0 and len_vert == 0:
        discs = list(Discs.keys())
        a = "Intervertebral disc: " + format_list(discs) + ".\nNo vertebrae showed abnormalities"
        qa_pairs.append((q, a))
    elif len_vert > 0 and len_disc == 0:
        verts = list(Vertbras.keys())
        a = "Vertebra: " + format_list(verts) + ".\nNo discs showed abnormalities"
        qa_pairs.append((q, a))
    return qa_pairs

def Disc_questions(Discs: Dict) -> List[Tuple[str, str]]:
    qa_pairs = []
    len_disc = len(Discs)
    if len_disc == 0:
        return qa_pairs

    for disc_name, disc_info in Discs.items():
        has_herniation = disc_info.get('herniation_type') is not None
        q_herniation = f"Does the {disc_name} intervertebral disc have herniation? If so, what is the type of herniation?"
        if has_herniation:
            a_herniation = f"Yes, the {disc_name} intervertebral disc has herniation. The type of herniation is {disc_info['herniation_type']}."
        else:
            a_herniation = f"No, the {disc_name} intervertebral disc does not have herniation."
        qa_pairs.append((q_herniation, a_herniation))

        if disc_info.get('disc_direction') is not None:
            q_direction = f"What is the direction of the herniation in the {disc_name} intervertebral disc?"
            a_direction = f"{disc_info['disc_direction']}"
            qa_pairs.append((q_direction, a_direction))

        if disc_info.get('additional_findings') and len(disc_info['additional_findings']) > 0:
            q_add = f"Are there any additional findings for the {disc_name} intervertebral disc?"
            a_add = format_list(disc_info['additional_findings'])
            qa_pairs.append((q_add, a_add))

        q5 = f"Describe the condition of the {disc_name} intervertebral disc."
        a5_parts = []

        if has_herniation:
            a5_parts.append(f"has {disc_info['herniation_type']} herniation")
            if disc_info.get('disc_direction'):
                a5_parts.append(f"with {disc_info['disc_direction']} direction type")

        if disc_info.get('additional_findings') and len(disc_info['additional_findings']) > 0:
            a5_parts.append(f"additional findings: {format_list(disc_info['additional_findings'])}")

        a5 = "; ".join(a5_parts) if a5_parts else "No significant abnormalities detected"
        if random.random() < 0.8:
            qa_pairs.append((q5, a5))
    return qa_pairs

def Vertbrae_questions(Vertbras: Dict) -> List[Tuple[str, str]]:
    qa_pairs = []
    len_vert = len(Vertbras)
    if len_vert == 0:
        return qa_pairs

    for vert_name, vert_info in Vertbras.items():
        findings = vert_info['vertebra_findings']

        if len(findings) > 0:
            has_fracture = "fracture" in findings
            q_fracture = f"Is the {vert_name} vertebra fractured?"
            if has_fracture:
                has_fracture_type = vert_info.get('fracture_type') is not None
                a_fracture = f"Yes, the {vert_name} vertebra is fractured."
                qa_pairs.append((q_fracture, a_fracture))
                if has_fracture_type:
                    q_fracture_type = f"What type of fracture is the {vert_name} vertebra?"
                    a_fracture_type = vert_info.get('fracture_type')
                    qa_pairs.append((q_fracture_type, a_fracture_type))
            else:
                a_fracture = f"No, the {vert_name} vertebra is not fractured."
                if random.random() < 0.15:
                    qa_pairs.append((q_fracture, a_fracture))

            if findings != ['fracture']:
                q_add = f"Is there any finding for the {vert_name} vertebra?"
                a_add = format_list(findings)
                qa_pairs.append((q_add, a_add))

        has_spondylolisthesis = vert_info.get('spondylolisthesis') is not None
        has_listhesis_direction = vert_info.get('listhesis_direction') is not None
        q_base = f"Is there spondylolisthesis in the vertebra {vert_name}? If so, what is the direction of the spondylolisthesis?"
        if has_listhesis_direction:
            if has_spondylolisthesis:
                if random.random() < 0.5:
                    q_slip = f"Is there spondylolisthesis in the vertebra {vert_name}? If yes, what are the direction and degree of the spondylolisthesis respectively?"
                    a_slip = f"Yes, the direction of the spondylolisthesis is {vert_info['listhesis_direction']} and the degree of the spondylolisthesis is {vert_info.get('spondylolisthesis')}."
                    qa_pairs.append((q_slip, a_slip))
                else:
                    a_base = f"Yes, the direction of the spondylolisthesis is {vert_info['listhesis_direction']}."
                    qa_pairs.append((q_base, a_base))
                    q_grade = f"What is the grade of the spondylolisthesis?"
                    a_grade = vert_info.get('spondylolisthesis')
                    qa_pairs.append((q_grade, a_grade))
            else:
                a_base = f"Yes, the direction of the spondylolisthesis is {vert_info['listhesis_direction']}."
        else:
            a_base = "No, there is no evidence of spondylolisthesis at vertebra {vert_name}.".format(vert_name=vert_name)
            if random.random() < 0.15:
                qa_pairs.append((q_base, a_base))
    return qa_pairs

def DegenerativeChanges_questions(DegenerativeChanges: Dict, Age: str) -> List[Tuple[str, str]]:
    qa_pairs = []
    age = str(Age).strip()
    include_age = bool(age) and age.lower() not in {"unknown", "none", "nan"}

    def maybe_with_age(question: str) -> str:
        if include_age:
            return f"The patient is {age} years old. {question}"
        return question

    q_general = "Does the patient have general lumbar degenerative changes?"
    if DegenerativeChanges.get('general_degeneration'):
        a_general = "Yes, the patient has general lumbar degenerative changes."
    else:
        a_general = "No, the patient does not have general lumbar degenerative changes."
    if random.random() < 0.5:
        qa_pairs.append((q_general, a_general))
    else:
        qa_pairs.append((maybe_with_age(q_general), a_general))

    len_vert = len(DegenerativeChanges['vertebra_degeneration'])
    len_disc = len(DegenerativeChanges['disc_degeneration'])
    q_disc = "Which intervertebral discs of the patient have degenerative changes?"
    q_vert = "Are there any vertebral segments with degeneration in this patient?"
    if len_vert > 0:
        a_vert = f"The patient has degenerative changes in the vertebrae {format_list(DegenerativeChanges['vertebra_degeneration'])}"
        if random.random() < 0.5:
            qa_pairs.append((q_vert, a_vert))
        else:
            qa_pairs.append((maybe_with_age(q_vert), a_vert))
    else:
        a_vert = "No, there are no vertebral segments showing degeneration in the patient."
        if random.random() < 0.15:
            if random.random() < 0.5:
                qa_pairs.append((q_vert, a_vert))
            else:
                qa_pairs.append((maybe_with_age(q_vert), a_vert))

    if len_disc > 0:
        a_disc = f"The patient has degenerative changes in the disc {format_list(DegenerativeChanges['disc_degeneration'])}"
        if random.random() < 0.5:
            qa_pairs.append((q_disc, a_disc))
        else:
            qa_pairs.append((maybe_with_age(q_disc), a_disc))
    else:
        a_disc = "No, there are no intervertebral discs showing degeneration in the patient."
        if random.random() < 0.15:
            if random.random() < 0.5:
                qa_pairs.append((q_disc, a_disc))
            else:
                qa_pairs.append((maybe_with_age(q_disc), a_disc))
    return qa_pairs

def OtherSystems_questions(OtherSystems: List) -> List[Tuple[str, str]]:
    qa_pairs = []
    if not OtherSystems:
        if random.random() < 0.05:
            q = "Are there any abnormalities in other systems (except for the lumbar spine)?"
            a = "No evidence of any abnormalities in other systems."
            qa_pairs.append((q, a))
        return qa_pairs

    q = "Are there any abnormalities in other systems (except for the lumbar spine)? If so, what are they?"
    a = format_list(OtherSystems)
    qa_pairs.append((q, a))
    return qa_pairs

def GeneralFindings_questions(GeneralFindings: List) -> List[Tuple[str, str]]:
    qa_pairs = []
    q = (
        "Please focus exclusively on conditions that do not have a specific anatomical "
        "location, including generalized or global lumbar spine conditions. What are they?"
    )
    a = format_list(GeneralFindings)
    if a != "":
        qa_pairs.append((q, a))
    return qa_pairs

def generate_qas4sample(content: Dict):
    qa_pairs = []
    position_qa = position_questions(content['Discs'], content['Vertbras'])
    qa_pairs.extend(position_qa)

    disc_qa = Disc_questions(content['Discs'])
    qa_pairs.extend(disc_qa)

    vert_qa = Vertbrae_questions(content['Vertbras'])
    qa_pairs.extend(vert_qa)

    degen_qa = DegenerativeChanges_questions(content['DegenerativeChanges'], content['Age'])
    qa_pairs.extend(degen_qa)

    other_qa = OtherSystems_questions(content['OtherSystems'])
    qa_pairs.extend(other_qa)

    gen_qa = GeneralFindings_questions(content['GeneralFindings'])
    qa_pairs.extend(gen_qa)
    return qa_pairs

def Description_questions(description: str) -> List[Tuple[str, str]]:
    qa_pairs = []
    if not description:
        return qa_pairs

    terminology_prompts = [
        "using appropriate radiological terminology, describe the MRI findings in detail.",
        "write a professional MRI description employing standard radiology lexicon.",
        "generate a description that uses precise medical terminology for the observed abnormalities.",
        "compose a radiology report description with accurate anatomical and pathological terms."
    ]
    qa_pairs.append((random.choice(terminology_prompts), description))
    return qa_pairs

def available_modalities(folder):
    modalities = []
    if os.path.exists(os.path.join(folder, "T1_slice.png")):
        modalities.append("T1")
    if os.path.exists(os.path.join(folder, "T2_slice.png")):
        modalities.append("T2")
    return modalities

def load_data(data_root, mode="full"):
    dirs = glob.glob(os.path.join(data_root, "*"))
    def check_folder_valid(folder):
        if not os.path.isdir(folder) or not os.path.exists(os.path.join(folder, "report.json")):
            return False
        modalities = available_modalities(folder)
        if mode == "partial":
            return len(modalities) > 0
        return {"T1", "T2"} <= set(modalities)
    valid_dirs = [dir for dir in dirs if check_folder_valid(dir)]
    samples = {}

    for dir in valid_dirs:
        with open(os.path.join(dir, "report.json"), "rb") as f:
            report = json.load(f)

        samples[dir] = report
    return samples

class PartialReport_Dataset(Dataset):
    def __init__(self, data_root):
        super().__init__()
        self.samples = load_data(data_root, mode="partial")
        self.messages = []

        for dir, report in self.samples.items():
            knowledge_qas= generate_qas4sample(report)
            for knowledge_qa in knowledge_qas:
                message, text = self.generate_message4knowledge_qa(dir, qa=knowledge_qa)
                self.messages.append({
                    "Message": message,
                    "Text": text,
                    "Patient_ID": dir,
                    "Task": "partial_qa"
                })
    def __len__(self):
        return len(self.messages)

    def __getitem__(self, index):
        return self.messages[index]

    def generate_message4knowledge_qa(self, dir, qa: Tuple[str, str]):
        sys_msg="You are a radiologist specializing in lumbar spine diseases."
        sys_dict = {"role":"system", "content": sys_msg}

        modalities = available_modalities(dir)
        if not modalities:
            raise FileNotFoundError(f"No T1_slice.png or T2_slice.png found in {dir}")
        modality_choice = random.choice(modalities)
        image = os.path.join(dir, f"{modality_choice}_slice.png")

        user_txt = f"The image is a slice of {modality_choice} weighted MRI sequences. Based on the slice, answer the following question. {qa[0]}"
        user_msg = [
            {"type":"image", "image": image, "resized_height": 448, "resized_width": 448},
            {"type": "text", "text": user_txt}
        ]
        user_dict = {
            "role": "user",
            "content": user_msg
        }
        return [sys_dict, user_dict], qa[1]

class FullReport_Dataset(Dataset):
    def __init__(self, data_root):
        super().__init__()
        self.samples = load_data(data_root, mode="full")
        self.messages = []
        for dir, report in self.samples.items():
            message, text = self.generate_sample_message(dir, report)
            des_qas = Description_questions(report['Description'])
            self.messages.append({
                "Message": message,
                "Text": text,
                "Patient_ID": dir,
                "Task": "report"
            })
            for des_qa in des_qas:
                message, text = self.generate_message4des_qa(dir, age=report['Age'], qa=des_qa)
                self.messages.append({
                    "Message": message,
                    "Text": text,
                    "Patient_ID": dir,
                    "Task": "description_qa"
                })
    def __len__(self):
        return len(self.messages)

    def __getitem__(self, index):
        return self.messages[index]

    def generate_message4des_qa(self, dir, age, qa: Tuple[str, str]):
        sys_msg="You are a radiologist specializing in lumbar spine diseases " \
            "and are good at writing structured and standardized radiology diagnostic reports based on MRI images."
        sys_dict = {"role":"system", "content": sys_msg}

        t1_image = os.path.join(dir, "T1_slice.png")
        t2_image = os.path.join(dir, "T2_slice.png")

        user_txt = f"This is a {age}-year-old patient. These two images are slices from T1-weighted and T2-weighted MRI sequences, respectively. Based on the information of these two images, {qa[0]}"
        user_msg = [
            {"type":"image", "image": t1_image, "resized_height": 448, "resized_width": 448},
            {"type":"image", "image": t2_image, "resized_height": 448, "resized_width": 448},
            {"type": "text", "text": user_txt}
        ]
        user_dict = {
            "role": "user",
            "content": user_msg
        }
        return [sys_dict, user_dict], qa[1]

    def generate_sample_message(self, dir, sample):
        sys_msg="You are a radiologist specializing in lumbar spine diseases " \
            "and are good at writing structured and standardized radiology diagnostic reports based on MRI images."
        sys_dict = {"role":"system", "content": sys_msg}

        t1_image = os.path.join(dir, "T1_slice.png")
        t2_image = os.path.join(dir, "T2_slice.png")
        age = sample['Age']

        user_txt = f"This is a {age}-year-old patient. These two images are slices from T1-weighted and T2-weighted MRI sequences, respectively. Based on the MRI information provided, please generate a professional and accurate MRI report in the format of Description and Impression."
        user_msg = [
            {"type":"image", "image": t1_image, "resized_height": 448, "resized_width": 448},
            {"type":"image", "image": t2_image, "resized_height": 448, "resized_width": 448},
            {"type": "text", "text": user_txt}
        ]
        user_dict = {
            "role": "user",
            "content": user_msg
        }
        answer = "Description: " + sample['Description'] + "\nImpression: " + sample['Impression']
        return [sys_dict, user_dict], answer

def make_dataset(data_root, mode):
    if mode == "full":
        return FullReport_Dataset(data_root)
    if mode == "partial":
        return PartialReport_Dataset(data_root)
    raise ValueError("mode should be full or partial")

def custom_collate_fn(batch):
    collated_batch = {
        "Message": [],
        "Patient_ID": [],
        "Text": [],
        "Task": []
    }
    for item in batch:
        collated_batch["Message"].append(item["Message"])
        collated_batch["Patient_ID"].append(item["Patient_ID"])
        collated_batch["Text"].append(item["Text"])
        collated_batch["Task"].append(item.get("Task", "unknown"))
    return collated_batch

def build_dataloader(data_root, batch_size, mode="full"):
    split_dirs = {split: os.path.join(data_root, split) for split in ("train", "val", "test")}
    if all(os.path.isdir(path) for path in split_dirs.values()):
        datasets = {split: make_dataset(path, mode) for split, path in split_dirs.items()}
        print(
            "[Info] Using predefined split folders. "
            f"Samples Train: {len(datasets['train'])}, Val: {len(datasets['val'])}, Test: {len(datasets['test'])}"
        )
        return {
            "train": DataLoader(
                datasets["train"],
                batch_size=batch_size,
                shuffle=True,
                collate_fn=custom_collate_fn,
            ),
            "val": DataLoader(
                datasets["val"],
                batch_size=batch_size,
                shuffle=False,
                collate_fn=custom_collate_fn,
            ),
            "test": DataLoader(
                datasets["test"],
                batch_size=batch_size,
                shuffle=False,
                collate_fn=custom_collate_fn,
            ),
        }

    dataset = make_dataset(data_root, mode)

    patient_ids = sorted({item["Patient_ID"] for item in dataset.messages})
    generator = torch.Generator().manual_seed(1234)
    permutation = torch.randperm(len(patient_ids), generator=generator).tolist()
    shuffled_patient_ids = [patient_ids[index] for index in permutation]

    num_patients = len(shuffled_patient_ids)
    num_train_patients = int(num_patients * 0.8)
    num_val_patients = int(num_patients * 0.1)
    train_patient_ids = set(shuffled_patient_ids[:num_train_patients])
    val_patient_ids = set(shuffled_patient_ids[num_train_patients:num_train_patients + num_val_patients])
    test_patient_ids = set(shuffled_patient_ids[num_train_patients + num_val_patients:])

    train_indices = [index for index, item in enumerate(dataset.messages) if item["Patient_ID"] in train_patient_ids]
    val_indices = [index for index, item in enumerate(dataset.messages) if item["Patient_ID"] in val_patient_ids]
    test_indices = [index for index, item in enumerate(dataset.messages) if item["Patient_ID"] in test_patient_ids]

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    print(f"[Info] Patients Train: {len(train_patient_ids)}, Val: {len(val_patient_ids)}, Test: {len(test_patient_ids)}")
    print(f"[Info] Samples Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=custom_collate_fn)
    return {'train': train_loader, 'val': val_loader, 'test': test_loader}
