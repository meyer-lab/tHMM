def Lin_shak(MASinitCells, MASlocBern, MAScGom, MASscaleGom, initCells2, locBern2, cGom2, scaleGom2):
    'Shakthis lineage where a second state is appended to first'
    MASexperimentTime = T_MAS
        masterLineage = gpt(MASexperimentTime, MASinitCells, MASlocBern, MAScGom, MASscaleGom)
        masterLineage = remove_NaNs(masterLineage)
        while len(masterLineage) == 0:
            masterLineage = gpt(MASexperimentTime, MASinitCells, MASlocBern, MAScGom, MASscaleGom)
            masterLineage = remove_NaNs(masterLineage)
        experimentTime2 = T_2
        sublineage2 = gpt(experimentTime2, initCells2, locBern2, cGom2, scaleGom2)
        sublineage2 = remove_NaNs(sublineage2)
        while len(sublineage2) == 0:
            sublineage2 = gpt(experimentTime2, initCells2, locBern2, cGom2, scaleGom2)
            sublineage2 = remove_NaNs(sublineage2)

        cell_endT_holder = []
        for cell in masterLineage:
            cell_endT_holder.append(cell.endT)

        master_cell_endT = max(cell_endT_holder) # get the longest tau in the list
        master_cell_endT_idx = np.argmax(cell_endT_holder) # get the idx of the longest tau in the lineage
        master_cell = masterLineage[master_cell_endT_idx] # get the master cell via the longest tau index

        for cell in sublineage2:
            cell.linID = master_cell.linID
            cell.gen += master_cell.gen
            cell.startT += master_cell_endT
            cell.endT += master_cell_endT

        master_cell.left = sublineage2[0]
        sublineage2[0].parent = master_cell
        newLineage = masterLineage + sublineage2
        
        
        X = remove_NaNs(newLineage)
        print(len(newLineage))