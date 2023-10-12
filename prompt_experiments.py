
import logging
from typing import List, Set
from gen_model_ids import GEN_MODEL_IDS
from datasets import load_dataset
from run_gen import run_gen

logger = logging.getLogger(__name__)


def run_prompt_exps(data_files: List[str], model_ids: list[str], icls: List[str]):
  for pf in data_files:
    dataset = load_dataset("semaj83/ioqm", data_files=pf)['train']
    prefix = pf.split("_")[0]
    for icl in icls:
      for model_id in model_ids:
        logger.info(f"running prompt experiment for model: {model_id}, context: {icl}, joined by: {prefix}")
        run_gen(
            model_id, 
            prompt_ds=dataset, 
            prompt_range=(0, 50), 
            save_root=f"/content/drive/MyDrive/ioqm_data/prompt_tests/{prefix}_joined_",
            icl=icl
        )
        

if __name__ == "__main__":
  prompt_exp_data_files = ['and_prompt_test.txt','comma_prompt_test.txt']
  icls = ["PROMPT", "PROMPT on a table."]
  run_prompt_exps(prompt_exp_data_files, GEN_MODEL_IDS, icls)
