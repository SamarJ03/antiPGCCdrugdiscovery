import json
import argparse
import os

class Prompter:
    def get_synthesize_task_prompt(self):
        args = self.args
        utWords, ltWords, utRules, ltRules = (
            args['utWords'], args['ltWords'], args['utRules'], args['ltRules']
        )
        rdkitNames = [
            'E-State', 'Functional Group', 'Molecular Topology', 'Fingerprint Based', 
            'Surface Area', 'Structural', 'Physiochemical'
        ]
        prompt_dict = {}

        prompt_dict['rdkit_big'] = {}
        prompt_dict['rdkit_small'] = {}
        prompt_dict['rdkit_big']['all'] = f"""Assume you are an experienced chemist and biologist. Please come up with {utRules} rules that you believe are crucial to predict if a molecule acts as an inhibitor towards Polyploid Giant Cancer Cells (PGCC). Each rule must be pertaining descriptors found in rdkit.Chem.Descriptors, and quantitative comparative (i.e. 'Anti-PGCC compounds have values greater than x for a certain descriptor', 'Anti-PGCC compounds have values between x and y for a certain descriptor', etc.). Do not explain and be concise and within {utWords} words."""
        prompt_dict['rdkit_small']['all'] = f"""Assume you are an experienced chemist and biologist. Please come up with {ltRules} rules that you believe are crucial to predict if a molecule acts as an inhibitor towards Polyploid Giant Cancer Cells (PGCC). Each rule must be pertaining descriptors found in rdkit.Chem.Descriptors, and quantitative comparative (i.e. 'Anti-PGCC compounds have values greater than x for a certain descriptor', 'Anti-PGCC compounds have values between x and y for a certain descriptor', etc.). Do not explain and be concise and within {utWords} words."""
        
        for item in rdkitNames:
            rdkitPrompt = ("Assume you are an experienced chemist and biologist. Please come up with {numRules} rules for {descType} descriptors in the rdkit.Chem.Descriptors package that you believe to be crucial in predicting if a certain molecule acts as an inhibitor towards Polyploid Giant Cancer Cells (PGCC). {specificPrompt} Do not explain and be concise, within {numWords} words.")            

            specPrompt=None
            if item=='E-State':
                specPrompt = 'These features include features like MaxEStateIndex and various EState_VSA measurements, pretty much any descriptor in the module that has "EState" in the name.'
            elif item=='Functional Group':
                specPrompt = 'These features track pharmacologically relevant groups that determine reactivity and biological activity. Pretty much only features that starts with "fr_".'
            elif item=='Molecular Topology':
                specPrompt = 'These features quantify fundamental connectivity patterns and graph-theoretical properties, such as BalabanJ, AvgIPC, BertzCT, HallKierAlpha, Ipc, and anything starting with "Chi" or "Kappa".'
            elif item=='Fingerprint Based':
                specPrompt = 'These features encode complex structural patterns into numerical representations. Pretty much all three FpDensityMorgan descriptors, along with any that start with "BCUT2D".'
            elif item=='Surface Area':
                specPrompt = 'These features quantify how molecules interface with their environment. These descriptors include LabuteASA along with all "PEOE_VSA", "SMR_VSA", and "SlogP_VSA" descriptors.'
            # elif item=='Structural': # unfinished
            #     specPrompt = 'These features provide detailed accounting of molecular building blocks and organization. These descriptors include FractionCSP3, '
            # elif item=='Physiochemical':
            #     specPrompt = 'These features represent core physical and chemical attributes governing molecular behavior. They include properties like molecular weight (MolWt), lipophilicity (MolLogP), polarity (TPSA), and electronic characteristics that influence how molecules interact with biological systems.'
            elif item=='Structural':
                specPrompt = "Here is a list of the features involved: ['FractionCSP3', 'NHOHCount', 'NOCount', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles', 'NumAliphaticRings', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumAromaticRings', 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms', 'NumRotatableBonds', 'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles', 'NumSaturatedRings', 'RingCount']."
            elif item=='Physiochemical':
                specPrompt = "Here is a list of the features: ['qed', 'MolWt', 'HeavyAtomMolWt', 'ExactMolWt', 'NumValenceElectrons', 'NumRadicalElectrons', 'MaxPartialCharge', 'MinPartialCharge', 'MaxAbsPartialCharge', 'MinAbsPartialCharge', 'TPSA', 'HeavyAtomCount', 'MolLogP', 'MolMR']."

            prompt_dict['rdkit_big'][item] = rdkitPrompt.format_map({
                'numRules':'a rule or multiple rules', 'descType':item, 'specificPrompt':'specPrompt' , 'numWords':str(utWords)
            })
            prompt_dict['rdkit_small'][item] = rdkitPrompt.format_map({
                'numRules':'a rule or multiple rules', 'descType':item, 'specificPrompt':'specPrompt', 'numWords':str(ltWords)
            })

        prompt_dict['ecfp4_big'] = f"""Assume you are an experienced chemist and biologist. Please come up with {utRules} rules pertaining ecfp4 fingerprint presence that you believe are crucial to predict if a molecule acts as an inhibitor towards Polyploid Giant Cancer Cells (PGCC). Each rule must be about the ecfp4 fingerprint presence of specific bits or substructures of molecules found in rdkit.Chem.AllChem.GetMorganFingerprintAsBitVect(radius=2). For example, 'Anti-PGCC compounds contain the substructures at the ECFP4 bit positions [18, 54, 105]', 'Anti-PGCC compounds contain the substructures at the ECFP4_274', etc. Do not explain, be concise and within {utWords} words."""
        prompt_dict['ecfp4_small'] = f"""Assume you are an experienced chemist and biologist. Please come up with {ltRules} rules pertaining ecfp4 fingerprint presence that you believe are crucial to predict if a molecule acts as an inhibitor towards Polyploid Giant Cancer Cells (PGCC). Each rule must be about the ecfp4 fingerprint presence of specific bits or substructures of molecules found in rdkit.Chem.AllChem.GetMorganFingerprintAsBitVect(radius=2). For example, 'Anti-PGCC compounds contain the substructures at the ECFP4 bit positions [18, 54, 105]', 'Anti-PGCC compounds contain the substructures at the ECFP4_274', etc. Do not explain, be concise and within {utWords} words."""

        prompt_dict['maccs_big'] = f"""Assume you are an experienced chemist and biologist. Please come up with {utRules} rules pertaining maccs fingerprint presence that you believe are crucial to predict if a molecule acts as an inhibitor towards Polyploid Giant Cancer Cells (PGCC). Each rule must be about the maccs fingerprint presence of specific bits or substructures of molecules found in rdkit.Chem.MACCSkeys. For example, 'Anti-PGCC compounds contain the substructures at the maccs bit positions [18, 54, 105]', 'Anti-PGCC compounds contain the substructures at the MACCS_274', etc. Do not explain, be concise and within {utWords} words."""
        prompt_dict['maccs_small'] = f"""Assume you are an experienced chemist and biologist. Please come up with {ltRules} rules pertaining maccs fingerprint presence that you believe are crucial to predict if a molecule acts as an inhibitor towards Polyploid Giant Cancer Cells (PGCC). Each rule must be about the maccs fingerprint presence of specific bits or substructures of molecules found in rdkit.Chem.MACCSkeys. For example, 'Anti-PGCC compounds contain the substructures at the maccs bit positions [18, 54, 105]', 'Anti-PGCC compounds contain the substructures at the MACCS_274', etc. Do not explain, be concise and within {utWords} words."""

        prompt_dict['metaFingerprints_big'] = f"""Assume you are an experienced chemist and biologist. Please come up with {utRules} rules pertaining maccs and ecfp4 fingerprint presence that you believe are crucial to predict if a molecule acts as an inhibitor towards Polyploid Giant Cancer Cells (PGCC). Each rule must be about the maccs/ecfp4 fingerprint presence of specific bits or substructures of molecules found in rdkit.Chem.MACCSkeys and rdkit.Chem.AllChem.GetMorganFingerprintAsBitVect(radius=2). For example, 'Anti-PGCC compounds contain the substructures at the maccs bit positions [18, 54, 105] and ECFP4 bit positions [42, 93, 201]', 'Anti-PGCC compounds contain the substructures at the MACCS_274', 'Anti-PGCC compounds contain the substructures at the ECFP4_274', etc. Do not explain, be concise and within {utWords} words."""
        prompt_dict['metaFingerprints_small'] = f"""Assume you are an experienced chemist and biologist. Please come up with {ltRules} rules pertaining maccs and ecfp4 fingerprint presence that you believe are crucial to predict if a molecule acts as an inhibitor towards Polyploid Giant Cancer Cells (PGCC). Each rule must be about the maccs/ecfp4 fingerprint presence of specific bits or substructures of molecules found in rdkit.Chem.MACCSkeys and rdkit.Chem.AllChem.GetMorganFingerprintAsBitVect(radius=2). For example, 'Anti-PGCC compounds contain the substructures at the maccs bit positions [18, 54, 105] and ECFP4 bit positions [42, 93, 201]', 'Anti-PGCC compounds contain the substructures at the MACCS_274', 'Anti-PGCC compounds contain the substructures at the ECFP4_274', etc. Do not explain, be concise and within {utWords} words."""

        return prompt_dict

    def get_inference_task_prompt(self):
        args = self.args
        utWords, ltWords, utRules, ltRules = (
            args['utWords'], args['ltWords'], args['utRules'], args['ltRules']
        )
        prompt_dict = {}

        prompt_dict['rdkit'] = {}
        prompt_dict['rdkit']['all'] = f"""Assume you are a very experienced chemist. In the following data, label 1 indicates that the molecule inhibits polyploid giant cancer cells (PGCCs), while label 0 means it does not. The following data also includes each molecules corresponding descriptor data from rdkit.Chem.Descriptors. Infer step-by-step to come up with {utRules} rules that directly relate molecular descriptor values to PGCC inhibition activity. Each rule must be pertaining descriptors found in rdkit.Chem.Descriptors, and quantitative comparative (i.e. 'Anti-PGCC compounds have values greater than x for a certain descriptor', 'Anti-PGCC compounds have values between x and y for a certain descriptor', etc.). Do not explain the rule and make it concise, within {utWords} words."""
        rdkitNames = [
            'E-State', 'Functional Group', 'Molecular Topology', 'Fingerprint Based', 
            'Surface Area', 'Structural', 'Physiochemical'
        ]
        for item in rdkitNames:
            specPrompt=None
            if item=='E-State':
                specPrompt = 'These features include features like MaxEStateIndex and various EState_VSA measurements, pretty much any descriptor in the module that has "EState" in the name.'
            elif item=='Functional Group':
                specPrompt = 'These features track pharmacologically relevant groups that determine reactivity and biological activity. Pretty much only features that starts with "fr_".'
            elif item=='Molecular Topology':
                specPrompt = 'These features quantify fundamental connectivity patterns and graph-theoretical properties, such as BalabanJ, AvgIPC, BertzCT, HallKierAlpha, Ipc, and anything starting with "Chi" or "Kappa".'
            elif item=='Fingerprint Based':
                specPrompt = 'These features encode complex structural patterns into numerical representations. Pretty much all three FpDensityMorgan descriptors, along with any that start with "BCUT2D".'
            elif item=='Surface Area':
                specPrompt = 'These features quantify how molecules interface with their environment. These descriptors include LabuteASA along with all "PEOE_VSA", "SMR_VSA", and "SlogP_VSA" descriptors.'
            # elif item=='Structural': # unfinished
            #     specPrompt = 'These features provide detailed accounting of molecular building blocks and organization. These descriptors include FractionCSP3, '
            # elif item=='Physiochemical':
            #     specPrompt = 'These features represent core physical and chemical attributes governing molecular behavior. They include properties like molecular weight (MolWt), lipophilicity (MolLogP), polarity (TPSA), and electronic characteristics that influence how molecules interact with biological systems.'
            elif item=='Structural':
                specPrompt = "Here is a list of the features involved: ['FractionCSP3', 'NHOHCount', 'NOCount', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles', 'NumAliphaticRings', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumAromaticRings', 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms', 'NumRotatableBonds', 'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles', 'NumSaturatedRings', 'RingCount']."
            elif item=='Physiochemical':
                specPrompt = "Here is a list of the features: ['qed', 'MolWt', 'HeavyAtomMolWt', 'ExactMolWt', 'NumValenceElectrons', 'NumRadicalElectrons', 'MaxPartialCharge', 'MinPartialCharge', 'MaxAbsPartialCharge', 'MinAbsPartialCharge', 'TPSA', 'HeavyAtomCount', 'MolLogP', 'MolMR']."

            prompt_dict['rdkit'][item] = f"""Assume you are a very experienced chemist. In the following data, label 1 indicates that the molecule inhibits polyploid giant cancer cells (PGCCs), while label 0 means it does not. The following data also includes each molecules corresponding {item} descriptor data from rdkit.Chem.Descriptors. {specPrompt} Infer step-by-step to come up with {ltRules} rules that directly relate a molecule's {item} descriptor value to PGCC inhibition activity. Each rule must be pertaining descriptors found in rdkit.Chem.Descriptors, and quantitative comparative (i.e. 'Anti-PGCC compounds have values greater than x for a certain descriptor', 'Anti-PGCC compounds have values between x and y for a certain descriptor', etc.). Do not explain the rule and make it concise, within {utWords} words."""

        prompt_dict['ecfp4'] = f"""Assume you are a very experienced chemist. In the following data, label 1 indicates that the molecule inhibits polyploid giant cancer cells (PGCCs), while label 0 means it does not. The following data also includes each molecules corresponding ecfp4 fingerprints. Infer step-by-step to come up with {utRules} rules that relate a molecules ecfp4 fingerprints presence of specific bits or substructures to PGCC inhibitory activity. Each rule must be about the ecfp4 fingerprint presence of specific bits or substructures of molecules found in rdkit.Chem.AllChem.GetMorganFingerprintAsBitVect(radius=2). For example, 'Anti-PGCC compounds contain the substructures at the ECFP4 bit positions [18, 54, 105]', 'Anti-PGCC compounds contain the substructures at the ECFP4_274', etc. Do not explain, be concise and within {utWords} words."""
        prompt_dict['maccs'] = f"""Assume you are a very experienced chemist. In the following data, label 1 indicates that the molecule inhibits polyploid giant cancer cells (PGCCs), while label 0 means it does not. The following data also includes each molecules corresponding maccs fingerprints. Infer step-by-step to come up with {utRules} rules that relate a molecules maccs fingerprints presence of specific bits or substructures to PGCC inhibitory activity. Each rule must be about the maccs fingerprint presence of specific bits or substructures of molecules found in rdkit.Chem.MACCSkeys. For example, 'Anti-PGCC compounds contain the substructures at the MACCS bit positions [18, 54, 105]', 'Anti-PGCC compounds contain the substructures at the MACCS_274', etc. Do not explain, be concise and within {utWords} words."""
        prompt_dict['metaFingerprints'] = f"""Assume you are a very experienced chemist. In the following data, label 1 indicates that the molecule inhibits polyploid giant cancer cells (PGCCs), while label 0 means it does not. The following data also includes each molecules corresponding maccs and ecfp4 fingerprints. Infer step-by-step to come up with {utRules} rules that relate a molecules maccs/ecfp4 fingerprints presence of specific bits or substructures to PGCC inhibitory activity. Each rule must be about the maccs/ecfp4 fingerprint presence of specific bits or substructures of molecules found in rdkit.Chem.MACCSkeys or rdkit.Chem.AllChem.GetMorganFingerprintAsBitVect(radius=2). For example, 'Anti-PGCC compounds contain the substructures at the MACCS bit positions [18, 54, 105] and ECFP4 bit positions [46, 104, 54]', 'Anti-PGCC compounds contain the substructures at the MACCS_274', 'Anti-PGCC compounds contain the substructures at the ECFP4_356', etc. Do not explain, be concise and within {utWords} words."""

        return prompt_dict

    def run(self):
        args = self.args
        if args['task'] == 'synthesize':
            prompt_dict = self.get_synthesize_task_prompt()
        elif args['task'] == 'inference':
            prompt_dict = self.get_inference_task_prompt()
        else:
            raise NotImplementedError(f"No implementation for task {args['task']}.")
        
        if not os.path.exists(args['output_folder']):
            os.makedirs(args['output_folder'])
        output_filename = os.path.join(args['output_folder'], f'{args["task"]}_prompt.json')
        with open(output_filename, 'w') as f:
            json.dump(prompt_dict, f, indent=2)

    def __init__(self, args=None, task=None, output=None):
        if args is None: 
            parser = argparse.ArgumentParser()
            parser.add_argument('--task', type=str, default='', help='synthesize/inference')
            parser.add_argument('--output_folder', type=str, default='prompt_file', help='prompt json file output folder')
            parser.add_argument('--ltWords', type=int, default=5, help='lower word limit')
            parser.add_argument('--utWords', type=int, default=20, help='upper word limit')
            parser.add_argument('--ltRules', type=int, default=3, help='lower token limit')
            parser.add_argument('--utRules', type=int, default=20, help='lower token limit')
            args = parser.parse_args()
        if isinstance(args, argparse.Namespace): args = vars(args)
        if isinstance(args, dict):
            if str(task) not in ('synthesize', 'inference'): raise Exception('invalid task for prompt creation..')
            else: args['task'] = str(task)
            if not output: raise Exception('No output folder provided..')
            else: args['output_folder'] = str(output)
        self.args = args