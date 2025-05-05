import json
import argparse
import os
from rdkit.Chem import Descriptors

class createPrompts:
    def __init__(self, args):
        self.args = args
        self.tokenizer, self.pipeline = self.get_hf_tokenizer_pipeline(args.model)
        self.system_prompt = self.get_system_prompt()
        self.prompt = self.get_inference_prompt()

    def get_synthesize_task_prompt(utWords, ltWords, utRules, ltRules):
        prompt_dict = {}

        prompt_dict['rdkit_big'] = {}
        prompt_dict['rdkit_small'] = {}
        prompt_dict['rdkit_big']['all'] = f"""Assume you are an experienced chemist and biologist. Please come up with {utRules} rules that you believe are crucial to predict if a molecule acts as an inhibitor towards Polyploid Giant Cancer Cells (PGCC). Each rule must be pertaining descriptors found in rdkit.Chem.Descriptors, and quantitative comparative (i.e. 'Anti-PGCC compounds have values greater than x for a certain descriptor', 'Anti-PGCC compounds have values between x and y for a certain descriptor', etc.). Do not explain and be concise and within {utWords} words."""
        prompt_dict['rdkit_small']['all'] = f"""Assume you are an experienced chemist and biologist. Please come up with {ltRules} rules that you believe are crucial to predict if a molecule acts as an inhibitor towards Polyploid Giant Cancer Cells (PGCC). Each rule must be pertaining descriptors found in rdkit.Chem.Descriptors, and quantitative comparative (i.e. 'Anti-PGCC compounds have values greater than x for a certain descriptor', 'Anti-PGCC compounds have values between x and y for a certain descriptor', etc.). Do not explain and be concise and within {utWords} words."""
        rdkitNames = [name for name, fxn in Descriptors.descList]
        for item in rdkitNames:
            rdkitPrompt = ("Assume you are an experienced chemist and biologist. Please come up with {numRules} to predict the value for the descriptor: {desc} (from rdkit.Chem.Descriptors), that you believe is crucial to predict if a certain molecule acts as an inhibitor towards Polyploid Giant Cancer Cells (PGCC). The rule must be quantitative and pertaining the {desc} descriptor of a molecule (i.e. 'An anti-PGCC compound has a(n) {desc} value greater than x', 'An anti-PGCC compound has a(n) {desc} value greater than y, but no less than z', 'An anti-PGCC compound has a(n) {desc} value if its abc values are over/under def', etc.). Do not explain and be concise, within {numWords} words.")
            prompt_dict['rdkit_big'][item] = rdkitPrompt.format_map({
                'numRules':'a rule or multiple rules', 'desc':item, 'numWords':str(utWords)
            })
            prompt_dict['rdkit_small'][item] = rdkitPrompt.format_map({
                'numRules':'a rule or multiple rules', 'desc':f'[{item}]', 'numWords':str(ltWords)
            })

        prompt_dict['ecfp4_big'] = f"""Assume you are an experienced chemist and biologist. Please come up with {utRules} rules pertaining ecfp4 fingerprint presence that you believe are crucial to predict if a molecule acts as an inhibitor towards Polyploid Giant Cancer Cells (PGCC). Each rule must be about the ecfp4 fingerprint presence of specific bits or substructures of molecules found in rdkit.Chem.AllChem.GetMorganFingerprintAsBitVect(radius=2). For example, 'Anti-PGCC compounds contain the substructures at the ECFP4 bit positions [18, 54, 105]', 'Anti-PGCC compounds contain the substructures at the ECFP4_274', etc. Do not explain, be concise and within {utWords} words."""
        prompt_dict['ecfp4_small'] = f"""Assume you are an experienced chemist and biologist. Please come up with {ltRules} rules pertaining ecfp4 fingerprint presence that you believe are crucial to predict if a molecule acts as an inhibitor towards Polyploid Giant Cancer Cells (PGCC). Each rule must be about the ecfp4 fingerprint presence of specific bits or substructures of molecules found in rdkit.Chem.AllChem.GetMorganFingerprintAsBitVect(radius=2). For example, 'Anti-PGCC compounds contain the substructures at the ECFP4 bit positions [18, 54, 105]', 'Anti-PGCC compounds contain the substructures at the ECFP4_274', etc. Do not explain, be concise and within {utWords} words."""

        prompt_dict['maccs_big'] = f"""Assume you are an experienced chemist and biologist. Please come up with {utRules} rules pertaining maccs fingerprint presence that you believe are crucial to predict if a molecule acts as an inhibitor towards Polyploid Giant Cancer Cells (PGCC). Each rule must be about the maccs fingerprint presence of specific bits or substructures of molecules found in rdkit.Chem.MACCSkeys. For example, 'Anti-PGCC compounds contain the substructures at the maccs bit positions [18, 54, 105]', 'Anti-PGCC compounds contain the substructures at the MACCS_274', etc. Do not explain, be concise and within {utWords} words."""
        prompt_dict['maccs_small'] = f"""Assume you are an experienced chemist and biologist. Please come up with {ltRules} rules pertaining maccs fingerprint presence that you believe are crucial to predict if a molecule acts as an inhibitor towards Polyploid Giant Cancer Cells (PGCC). Each rule must be about the maccs fingerprint presence of specific bits or substructures of molecules found in rdkit.Chem.MACCSkeys. For example, 'Anti-PGCC compounds contain the substructures at the maccs bit positions [18, 54, 105]', 'Anti-PGCC compounds contain the substructures at the MACCS_274', etc. Do not explain, be concise and within {utWords} words."""

        prompt_dict['combinedFingerprints_big'] = f"""Assume you are an experienced chemist and biologist. Please come up with {utRules} rules pertaining maccs and ecfp4 fingerprint presence that you believe are crucial to predict if a molecule acts as an inhibitor towards Polyploid Giant Cancer Cells (PGCC). Each rule must be about the maccs/ecfp4 fingerprint presence of specific bits or substructures of molecules found in rdkit.Chem.MACCSkeys and rdkit.Chem.AllChem.GetMorganFingerprintAsBitVect(radius=2). For example, 'Anti-PGCC compounds contain the substructures at the maccs bit positions [18, 54, 105] and ECFP4 bit positions [42, 93, 201]', 'Anti-PGCC compounds contain the substructures at the MACCS_274', 'Anti-PGCC compounds contain the substructures at the ECFP4_274', etc. Do not explain, be concise and within {utWords} words."""
        prompt_dict['combinedFingerprints_small'] = f"""Assume you are an experienced chemist and biologist. Please come up with {ltRules} rules pertaining maccs and ecfp4 fingerprint presence that you believe are crucial to predict if a molecule acts as an inhibitor towards Polyploid Giant Cancer Cells (PGCC). Each rule must be about the maccs/ecfp4 fingerprint presence of specific bits or substructures of molecules found in rdkit.Chem.MACCSkeys and rdkit.Chem.AllChem.GetMorganFingerprintAsBitVect(radius=2). For example, 'Anti-PGCC compounds contain the substructures at the maccs bit positions [18, 54, 105] and ECFP4 bit positions [42, 93, 201]', 'Anti-PGCC compounds contain the substructures at the MACCS_274', 'Anti-PGCC compounds contain the substructures at the ECFP4_274', etc. Do not explain, be concise and within {utWords} words."""

        return prompt_dict

    def get_inference_task_prompt(utWords, ltWords, utRules, ltRules):
        prompt_dict = {}

        prompt_dict['rdkit'] = {}
        prompt_dict['rdkit']['all'] = f"""Assume you are a very experienced chemist. In the following data, label 1 indicates that the molecule inhibits polyploid giant cancer cells (PGCCs), while label 0 means it does not. The following data also includes each molecules corresponding descriptor data from rdkit.Chem.Descriptors. Infer step-by-step to come up with {utRules} rules that directly relate molecular descriptor values to PGCC inhibition activity. Each rule must be pertaining descriptors found in rdkit.Chem.Descriptors, and quantitative comparative (i.e. 'Anti-PGCC compounds have values greater than x for a certain descriptor', 'Anti-PGCC compounds have values between x and y for a certain descriptor', etc.). Do not explain the rule and make it concise, within {utWords} words."""
        rdkitNames = [name for name, fxn in Descriptors.descList]
        for item in rdkitNames:
            prompt_dict['rdkit'][item] = f"""Assume you are a very experienced chemist. In the following data, label 1 indicates that the molecule inhibits polyploid giant cancer cells (PGCCs), while label 0 means it does not. The following data also includes each molecules corresponding {item} descriptor data from rdkit.Chem.Descriptors. Infer step-by-step to come up with {ltRules} rules that directly relate a molecule's {item} descriptor value to PGCC inhibition activity. Each rule must be pertaining descriptors found in rdkit.Chem.Descriptors, and quantitative comparative (i.e. 'Anti-PGCC compounds have values greater than x for a certain descriptor', 'Anti-PGCC compounds have values between x and y for a certain descriptor', etc.). Do not explain the rule and make it concise, within {utWords} words."""

        prompt_dict['ecfp4'] = f"""Assume you are a very experienced chemist. In the following data, label 1 indicates that the molecule inhibits polyploid giant cancer cells (PGCCs), while label 0 means it does not. The following data also includes each molecules corresponding ecfp4 fingerprints. Infer step-by-step to come up with {utRules} rules that relate a molecules ecfp4 fingerprints presence of specific bits or substructures to PGCC inhibitory activity. Each rule must be about the ecfp4 fingerprint presence of specific bits or substructures of molecules found in rdkit.Chem.AllChem.GetMorganFingerprintAsBitVect(radius=2). For example, 'Anti-PGCC compounds contain the substructures at the ECFP4 bit positions [18, 54, 105]', 'Anti-PGCC compounds contain the substructures at the ECFP4_274', etc. Do not explain, be concise and within {utWords} words."""
        prompt_dict['maccs'] = f"""Assume you are a very experienced chemist. In the following data, label 1 indicates that the molecule inhibits polyploid giant cancer cells (PGCCs), while label 0 means it does not. The following data also includes each molecules corresponding maccs fingerprints. Infer step-by-step to come up with {utRules} rules that relate a molecules maccs fingerprints presence of specific bits or substructures to PGCC inhibitory activity. Each rule must be about the maccs fingerprint presence of specific bits or substructures of molecules found in rdkit.Chem.MACCSkeys. For example, 'Anti-PGCC compounds contain the substructures at the MACCS bit positions [18, 54, 105]', 'Anti-PGCC compounds contain the substructures at the MACCS_274', etc. Do not explain, be concise and within {utWords} words."""
        prompt_dict['fingerprints'] = f"""Assume you are a very experienced chemist. In the following data, label 1 indicates that the molecule inhibits polyploid giant cancer cells (PGCCs), while label 0 means it does not. The following data also includes each molecules corresponding maccs and ecfp4 fingerprints. Infer step-by-step to come up with {utRules} rules that relate a molecules maccs/ecfp4 fingerprints presence of specific bits or substructures to PGCC inhibitory activity. Each rule must be about the maccs/ecfp4 fingerprint presence of specific bits or substructures of molecules found in rdkit.Chem.MACCSkeys or rdkit.Chem.AllChem.GetMorganFingerprintAsBitVect(radius=2). For example, 'Anti-PGCC compounds contain the substructures at the MACCS bit positions [18, 54, 105] and ECFP4 bit positions [46, 104, 54]', 'Anti-PGCC compounds contain the substructures at the MACCS_274', 'Anti-PGCC compounds contain the substructures at the ECFP4_356', etc. Do not explain, be concise and within {utWords} words."""

        return prompt_dict

    def run(self, args):
        if args.task == 'synthesize':
            prompt_dict = self.get_synthesize_task_prompt(args.utWords, args.ltWords, args.utRules, args.ltRules)
        elif args.task == 'inference':
            prompt_dict = self.get_inference_task_prompt(args.utWords, args.ltWords, args.utRules, args.ltRules)
        else:
            raise NotImplementedError(f"No implementation for task {args.task}.")
        
        if not os.path.exists(args.output_folder):
            os.makedirs(args.output_folder)
        output_filename = os.path.join(args.output_folder, f'{args.task}_prompt.json')
        with open(output_filename, 'w') as f:
            json.dump(prompt_dict, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='synthesize', help='synthesize/inference')
    parser.add_argument('--output_folder', type=str, default='LLM4SD_PGCC/prompt_file', help='prompt json file output folder')
    parser.add_argument('--ltWords', type=int, default=5, help='lower token limit')
    parser.add_argument('--utWords', type=int, default=20, help='upper token limit')
    parser.add_argument('--ltRules', type=int, default=3, help='lower token limit')
    parser.add_argument('--utRules', type=int, default=20, help='lower token limit')

args = parser.parse_args()
creator = createPrompts(args)
creator.run(args)
