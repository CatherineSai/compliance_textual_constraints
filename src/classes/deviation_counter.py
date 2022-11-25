
import numpy as np
import pandas as pd


class Deviation_Counter:
  ''' checks if the similarity scores are above/below threshold, specific combinations resulting in a deviation flag for subject/verb/object ''' 

  def __init__(self, nlp, gamma_one, gamma_two, gamma_three):
    self.nlp = nlp
    self.gamma_one = gamma_one
    self.gamma_two = gamma_two
    self.gamma_three = gamma_three

  def get_deviation_flag(self, df):
    '''calculates all phrase similarity pairs between reg and rea sentence pair'''
    # fill empty fields with numpy NaN (makes it easier for if statements to check if both are npt empty)
    df = df.replace(r'^\s*$', np.nan, regex=True)
    # for each row extract highest subject phrase score (4 possible)
    df["max_sub_sim"] = df[["sub_1_1_sim", "sub_1_2_sim", "sub_2_1_sim", "sub_2_2_sim"]].max(axis=1)
    # for each row extract highest subject phrase score (4 possible)
    df["max_verb_sim"] = df[["verb_1_1_sim", "verb_1_2_sim", "verb_2_1_sim", "verb_2_2_sim"]].max(axis=1)
    # for each row extract highest subject phrase score (4 possible)
    df["max_obj_sim"] = df[["obj_1_1_sim", "obj_1_2_sim", "obj_2_1_sim", "obj_2_2_sim"]].max(axis=1)

    # check if highest score sub/verb/obj below threshold & highest scores of the other two above threshold - if yes, raise deviation flag
    df['sub_deviation_exists'] = [1 if ((x < self.gamma_one) & (y >= self.gamma_two) & (z >= self.gamma_three)) else 0 for x,y,z in zip(df['max_sub_sim'],df['max_verb_sim'],df['max_obj_sim'])]
    df['verb_deviation_exists'] = [1 if ((x >= self.gamma_one) & (y < self.gamma_two) & (z >= self.gamma_three)) else 0 for x,y,z in zip(df['max_sub_sim'],df['max_verb_sim'],df['max_obj_sim'])]
    df['obj_deviation_exists'] = [1 if ((x >= self.gamma_one) & (y >= self.gamma_two) & (z < self.gamma_three)) else 0 for x,y,z in zip(df['max_sub_sim'],df['max_verb_sim'],df['max_obj_sim'])]
    # if sub/verb/obj are similar above threshold but one constraint includes a negation and the other doesn't
    df['neg_deviation_exists'] = [1 if ((x >= 0.7) & (y >= 0.5) & (z >= 0.8) & (n != 0)) else 0 for x,y,z,n in zip(df['max_sub_sim'],df['max_verb_sim'],df['max_obj_sim'],df['difference_in_negations'])] 
    # if more than one phrase is below the similarity threshold, the sentences seem to be to different after all to compare them - in this case they are counted as 'sent_sim_not_sufficient'
    df['sent_sim_not_sufficient'] = [1 if (((x < self.gamma_one) & (y < self.gamma_two)) or ((x < self.gamma_one) & (z < self.gamma_three)) or ((y < self.gamma_two) & (z < self.gamma_three))) else 0 for x,y,z in zip(df['max_sub_sim'],df['max_verb_sim'],df['max_obj_sim'])]
    return df

  def aggreagte_deviation_count(self, df, count_unmapped_reg_sent, count_unmapped_rea_sent, count_mapped_rea_sent):
    '''counts the number of deviations detected in each category. Result is a final outcome overview table'''
    result_overview_df = pd.DataFrame(columns=['case_1_number_constraints_reg_mapped','case_2_number_constraints_reg_unmapped','case_3_number_constraints_rea_unmapped','in_case_1_number_constraints_rea_mapped','gamma_one','gamma_two','gamma_three','number_subject_deviations','number_verb_deviations','number_object_deviations','number_negation_deviations','sent_mapped_but_sim_not_sufficient'])
    result_overview_df.at['case_1_number_constraints_reg_mapped'] = df['original_sentence_reg'].count()
    result_overview_df.at['case_2_number_constraints_reg_unmapped'] = count_unmapped_reg_sent
    result_overview_df.at['case_3_number_constraints_rea_unmapped'] = count_unmapped_rea_sent
    result_overview_df.at['in_case_1_number_constraints_rea_mapped'] = count_mapped_rea_sent
    result_overview_df.at['gamma_one'] = self.gamma_one
    result_overview_df.at['gamma_two'] = self.gamma_two
    result_overview_df.at['gamma_three'] = self.gamma_three
    result_overview_df.at['number_subject_deviations'] = df['sub_deviation_exists'].sum()
    result_overview_df.at['number_verb_deviations'] = df['verb_deviation_exists'].sum()
    result_overview_df.at['number_object_deviations'] = df['obj_deviation_exists'].sum()
    result_overview_df.at['number_negation_deviations'] = df['neg_deviation_exists'].sum()
    result_overview_df.at['sent_mapped_but_sim_not_sufficient'] = df['sent_sim_not_sufficient'].sum()
    return result_overview_df