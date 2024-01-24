from intellimed.Trainer import infer_bert

text = "Characterisation of the in vitro activity of the depsipeptide histone deacetylase inhibitor spiruchostatin A.  We recently completed the total synthesis of spiruchostatin A, a depsipeptide natural product with close structural similarities to FK228, a histone deacetylase (HDAC) inhibitor (HDI) currently being evaluated in clinical trials for cancer. Here we report a detailed characterisation of the in vitro activity of spiruchostatin A. Spiruchostatin A was a potent (sub-nM) inhibitor of class I HDAC activity in vitro and acted as a prodrug, requiring reduction for activity. Spiruchostatin A was a potent (low nM) inhibitor of the growth of various cancer cell lines. Spiruchostatin A-induced acetylation of specific lysine residues within histones H3 and H4, and increased the expression of p21(cip1/waf1), but did not induce acetylation of alpha-tubulin. Spiruchostatin A also induced cell cycle arrest, differentiation and cell death in MCF7 breast cancer cells. Like FK228, spiruchostatin A was both an inducer and substrate of the ABCB1 drug efflux pump. Whereas spiruchostatin A and FK228-induced protracted histone acetylation, hydroxamate HDI-induced short-lived histone acetylation. Using a subset of HDI-target genes identified by microarray analysis, we demonstrated that these differences in kinetics of histone acetylation between HDI correlated with differences in the kinetics of induction or repression of specific target genes. Our results demonstrate that spiruchostatin A is a potent inhibitor of class I HDACs and anti-cancer agent. Differences in the kinetics of action of HDI may be important for the clinical application of these compounds."

file_name_or_path = "E:\FNii\IE-Toolchain\pretrained\BioLinkBERT-base"

tokens, tags = infer_bert(text, file_name_or_path)

flag = False
entity = ""
entity_tag = ""
for token, tag in zip(tokens, tags):
    if flag == False:  # 代表当前是没有实体的
        if tag.startswith("B-"):
            entity = token
            entity_tag = tag[2:]
            flag = True
    elif flag == True:
        if tag.startswith("I-"):
            if token.startswith("##"):
                entity = entity + token[2:]
            else:
                entity = entity + " " + token
        else:
            print(entity_tag + ":  " + entity)
            if tag.startswith("O"):
                entity = ""
                entity_tag = ""
                flag = False
            elif tag.startswith("B-"):
                entity = token
                entity_tag = tag[2:]
                flag = True
