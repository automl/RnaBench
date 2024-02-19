from typing import Optional
import numpy as np
import RNA
import sys
import random
import subprocess
import re
import decimal
import math
import os
import shutil
import time
import types
import uuid
import argparse
from argparse import RawTextHelpFormatter, Namespace
import networkx as nx
from scipy.interpolate import *

##############################
# Additional Functionality for RnaBench
##############################


class AntaRNA():
    def __init__(self,
                 structure: Optional[str] = None):
        if structure:
            self._target = structure

    def __name__(self):
        return 'AntaRNA'

    def design(self,
               structure: Optional[str] = None,
               ):

        if structure is not None:
            self._target = structure

        args = Namespace(Cstr=self._target,
                           Cgcweight=5.0,
                           ConvergenceCount=130,
                           Cseq=None,
                           Cseqweight=1.0,
                           Cstrweight=0.5,
                           ER=0.2,
                           Resets=5,
                           alpha=1.0,
                           antsTerConv=50,
                           ants_per_selection=10,
                           beta=1.0,
                           improve_procedure='s',
                           level=1,
                           name='antaRNA',
                           noGUBasePair=False,
                           noLBPmanagement=True,
                           noOfColonies=1,
                           omega=2.23,
                           output_file='STDOUT',
                           output_verbose=False,
                           paramFile='',
                           plot=False,
                           py=False,
                           seed=None,
                           subparser_name=None,
                           tGC=[0.5, 0.5, 0.5],
                           tGCmax=-1.0,
                           tGCvar=-1.0,
                           temperature=37.0,
                           time=600,
                           verbose=False
                           )

        hill = AntHill()
        hill.params.readArgParseArguments(args)
        hill.params.py = False
        hill.params.check()





        if hill.params.error == "0":
            hill.swarm()
        else:
            print(hill.params.error)



        return ''.join(design)

    def __call__(self, structure):
        return self.design(structure=structure)





##############################
# ARGPARSE TYPES AND FUNCTIONS
##############################

def GC(string):
	"""
		argparse type definition of GC value
	"""
	## print ">"+string+"<"
	m = re.match('^\s*(\d+\.\d+):?(\d+)?\s*?-?\s*?(\d+)?\s*?$', string)

	if m is None:
		# print "'" + string + "' is not a valid input for tGC Expected forms like '0-5'."
		exit(1)
	tGC = float(m.group(1))

	if m.group(2) and m.group(3):

		start = int(m.group(2))
		end = int(m.group(3))
		## print "tGC", tGC, "Start", start, "End", end
		return (tGC, start, end)
	else:
		## print "tGC", tGC
		return (tGC)



def FuzzyConstraintFeature(string):
	"""
		argparse type definition of a fuzzy structure constraint feature
	"""
	m = re.match('^\s*([.a-zA-Z]*)\s*(\w+)\s*(\d+\.?\d+)\s*$', string)
	if m is None:
		# print "'" + string + "' is not a valid FuzzyConstraintFeature input of type STRUCTUREBLOCK STRING FLOAT"
		exit(1)
	s1 = m.group(1)
	s2 = m.group(2)
	s3 = float(m.group(3))
	return ( (s1, s2, s3) )



def DiffFuzzyConstraintFeature(string):
	"""
		argparse type definition of a differential fuzzy structure constraint feature
	"""
	m = re.match('^\s*([.a-zA-Z]*)\s*(\w+)\s*(\d+\.?\d+)\s*(\w+)\s*(\d+\.?\d+)\s*$', string)
	if m is None:
		# print "'" + string + "' is not a valid DiffFuzzyConstraintFeature input of type STRUCTUREBLOCK STRING FLOAT STRING FLOAT"
		exit(1)
	s1 = m.group(1)
	s2 = m.group(2)
	s3 = float(m.group(3))
	s4 = m.group(4)
	s5 = float(m.group(5))
	return ( (s1, s2, s3, s4, s5) )

def AccuracyFeature(string):
	"""
		argparse type definition of accuracy feature
	"""
	m = re.match('^\s*([.()]*)\s*(\w+)\s*(\d+\.?\d+)\s*$', string)
	if m is None:
		# print "'" + string + "' is not a valid AccuracyFeature input of type STRUCTURE STRING STRING/FLOAT"
		exit(1)
	s1 = m.group(1)
	s2 = m.group(2)
	s3 = float(m.group(3))
	## print (s1, s2, s3)
	return ( (s1, s2, s3) )

def AccessibilityFeature(string):
	"""
		argparse type definition of accessibility feature
	"""
	m = re.match('^\s*([.x]*)\s*(\w+)\s*(\d+\.?\d+)\s*$', string)
	if m is None:
		# print "'" + string + "' is not a valid AccessibilityFeature input of type STRUCTURE STRING STRING/FLOAT"
		exit(1)
	s1 = m.group(1)
	s2 = m.group(2)
	s3 = float(m.group(3))
	if len(s1.replace(".", "")) != len(s1.strip(".")):
		# print "defined accessibility", s1, "is not within one stretch! Please reconsider."
		exit(1)

	return ( (s1, s2, s3) )

def DiffAccuracyFeature(string):
	"""
		argparse type definition of differential accuracy feature
	"""
	m = re.match('^\s*([.()]*)\s*(\w+)\s*(\d+\.?\d+)\s*(\w+)\s*(\d+\.?\d+)\s*$', string)
	if m is None:
		# print "'" + string + "' is not a valid DiffAccuracyFeature input of type STRUCTURE STRING FLOAT STRING FLOAT"
		exit(1)
	s1 = m.group(1)
	s2 = m.group(2)
	s3 = float(m.group(3))
	s4 = m.group(4)
	s5 = float(m.group(5))
	return ( (s1, s2, s3, s4, s5) )

def DiffAccessibilityFeature(string):
	"""
		argparse type definition of differential accessibility feature
	"""
	m = re.match('^\s*([.x]*)\s*(\w+)\s*(\d+\.?\d+)\s*(\w+)\s*(\d+\.?\d+)\s*$', string)
	if m is None:
		# print "'" + string + "' is not a valid DiffAccessibilityFeature input of type STRUCTURE STRING FLOAT STRING FLOAT"
		exit(1)
	s1 = m.group(1)
	s2 = m.group(2)
	s3 = float(m.group(3))
	s4 = m.group(4)
	s5 = float(m.group(5))

	if len(s1.replace(".", "")) != len(s1.strip(".")):
		# print "defined diff_accessibility", s1, "is not within one stretch! Please reconsider."
		exit(1)
	return ( (s1, s2, s3, s4, s5) )

def parseGC(string):
	"""
		Argparse Type Function
		parses a string of "FLOAT:INT-INT", where ":INT-INT" is set to be optional.
		Returns a single value or a triple (FLAOT, INT, INT).
	"""
	m = re.match('^\s*(\d+\.\d+):?(\d+)?\s*?-?\s*?(\d+)?\s*?$', string)
	if m is None:
		# print "'" + string + "' is not a valid input for tGC Expected forms like '0-5'."
		exit(1)
	tGC = float(m.group(1))
	if m.group(2) and m.group(3):
		start = int(m.group(2))
		end = int(m.group(3))
		return (tGC, start, end)
	else:
		return (tGC)

def convert_arg_line_to_args(arg_line):
	"""
		argparse function to parse the variables
	"""
	for arg in arg_line.split():
		if not arg.strip():
			continue
		yield arg

#####################
# SUPPORT FUNCTIONS
#####################

def isfloat(value):
	"""
		checks type of float status
	"""
	try:
		float(value)
		return True
	except ValueError:
		return False


def print2file(f, i, m):
	"""
		# print content i to file f in mode m
	"""
	line = str(i)
	if m == "a":
		call = "echo \"" +  line + "\" >> " + f
	elif m == "w":
		call = "echo \"" +  line + "\" > " + f
	os.system(call)

def getUsedTime(start_time):
	"""
		Return the used time between -start time- and now.
	"""
	end_time = time.time()
	return end_time - start_time

def substr(x, string, subst):
	"""
		Classical substring function
	"""
	s1 = string[:x-1]

	s2 = string[x-1:x]
	s3 = string[x:]
  #s2 = s[x+len(string)-x-1:]

	return s1 + subst + s3

############################################
# STRUCTURE CHECKS
############################################

def isStructureCompatible(lp1, lp2 ,bp):
	"""
		Checks, if the region within lp1 and lp2 is structurally balanced
	"""
	x = lp1 + 1
	while (x < lp2):
		if (bp[x] <= lp1 or bp[x] > lp2):
			return False
		if x == bp[x]:
			x += 1
		else:
			x = bp[x] + 1
	return x == lp2
####################################
### EXTENDED CONSTRAINT MANAGEMENT
####################################
def getBPPM(sequence, structure = "", bppm_cutoff = 0.00001):
    """
        Requires ViennaRNAtools Python module
        Returns the base pair probability matrix using Vienna pf_fold, get_pr and free_pf_arrays functions.
        returns upper triangular matrix, whose entries exceed a threshold
    """
    bppm = {}


     #'--noPS', '-d 2', t, P



    if structure != "":
        RNA.cvar.fold_constrained = 1
    else:
        RNA.cvar.fold_constrained = 0
    ## print "Before", structure
    RNA.pf_fold(sequence, structure)
    ## print "After", structure
    seq_len = len(sequence)+1
    for i in range(1, seq_len):
        for j in range(1, seq_len):
            if i<j:
                bpp = RNA.get_pr(i,j)
                if bpp > bppm_cutoff:
                    bppm[str(i) + "_" + str(j)] = bpp
                else:
                    bppm[str(i) + "_" + str(j)] = 0
    RNA.free_pf_arrays()
    ## print bppm
    #exit(1)
    return bppm


def getAccuracy(struct_stack, bppm):
    """
        Calculate average structuredness of given structure(stack) within bppm
    """
    acc = 0
    for sq_i in struct_stack.keys():
        v = str(sq_i) + "_" + str(struct_stack[sq_i])
        if v in bppm:
            acc += bppm[v]
            #acc += math.pow(bppm[v], 2) / len(struct_stack)
    return acc


def getAccessibility(positions, l, bppm):
    """
        Calculate average unstructuredness of given positions within bppm of a sequence with length l
    """
    acc = 0
    for sq_i in positions.keys():
        #visits = 0
        #vis = []
        for i in range(1, sq_i ):
            #visits += 1
            #vis.append(i)
            v = str(i) + "_" + str(sq_i)
            if v in bppm:
                if bppm[v] > 0:

                    #val = math.pow(bppm[v], 2)
                    val = bppm[v]
                    acc += val
        for i in range(sq_i + 1, l + 1):
            #visits += 1
            #vis.append(i)
            v = str(sq_i) + "_" + str(i)
            if v in bppm:
                if bppm[v] > 0:
                    #val = math.pow(bppm[v], 2)
                    val = bppm[v]
                    acc += val
        ## print "Visits", visits
        ## print vis
   # # print "used positions", len(positions)
    return 1 -(acc/len(positions))




def removePairedAndUndefined_From_bpstack(P, struct_stack):
	"""
		Produce stack for the calculation of accessibility
	"""
	tmp_struct_stack = {}
	for i in struct_stack.keys():
		if struct_stack[i] == i and P[i] == "x":
			tmp_struct_stack[i + 1] = struct_stack[i] + 1
	return tmp_struct_stack

def removeUnpairedFrom_bpstack(struct_stack):
	"""
		Produce stack for the calculation of accuracy
	"""
	tmp_struct_stack = {}
	for i in struct_stack.keys():
		if struct_stack[i] > i:
			tmp_struct_stack[i + 1] = struct_stack[i] + 1
	return tmp_struct_stack

def all_indices(value, qlist):
    indices = []
    idx = -1
    while True:
        try:
            idx = qlist.index(value, idx+1)
            indices.append(idx)
        except ValueError:
            break
    return indices


def retrieveFuzzyStack(P):
	# # print P

	# Find out interaction type
	interaction_type = ""
	p = P.replace(".", "")
	## print p, p[0]
	if p[0].isupper():
		interaction_type = "self"
		P = "".join(([l.upper() for l in list(P)]))
	else:
		interaction_type = "non-self"
		P = "".join(([l.lower() for l in list(P)]))

	# Find out block types
	p = P.strip(".")
	block_type = ""
	if len(P.replace(".", "")) == len(p):
		block_type = "consecutive"

	else:
		block_type = "split"
	## print block_type, interaction_type
	# Check for a single block which just should be non self interacting
	if interaction_type == "non-self" and block_type == "consecutive":
		P = str([l.upper() for l in list(P)])
		interaction_type = "self"


	# retrieving the indices
	indices = all_indices(p[0], list(P))
	all_pairs_involved = []
	for i in indices:
		for j in indices:
			if abs(i-j) > 3 and i < j:
				all_pairs_involved.append((i,j))


	i = indices[0]
	j = ""
	jpk = ""
	l = ""
	prev = i
	for ri in range(1, len(indices)):
		if indices[ri] == prev + 1:
			prev = indices[ri]
		else:
			j = prev
			jpk = indices[ri]
			l = indices[-1]
			break
	## print i, j, jpk, l

	## print block_type, interaction_type
	if block_type == "split" and interaction_type == "non-self":
		dels = []
		for bp in all_pairs_involved:
			a, b = bp
			if ((i <= a and a <= j) and (i <= b and b <= j)) or ((jpk <= a and a <= l) and (jpk <= b and b <= l)) :# both indices are wihtin first block
				dels.append(bp)
		for d in dels:
			del all_pairs_involved[all_pairs_involved.index(d)]

	## print P
	## print indices
	## print all_pairs_involved
	#exit(1)
	return all_pairs_involved




def getStructStacks(structure_queries):
	"""
		for each structure query, produce appropriate stack
	"""
	s_q = []
	for ask_struct, constraint_system, type_of_measurement, optimi_arg in structure_queries:
		if type_of_measurement == "accuracy":
			s_q.append( (ask_struct,
						removeUnpairedFrom_bpstack(getbpStack(ask_struct)[0]),
						constraint_system,
						type_of_measurement,
						optimi_arg) )
		elif type_of_measurement == "access":
			s_q.append( (ask_struct,
						removePairedAndUndefined_From_bpstack(ask_struct , getbpStack(len(ask_struct) * ".") [0]),
						constraint_system,
						type_of_measurement,
						optimi_arg) )

	return s_q


############################################
# IUPAC LOADINGS AND NUCLEOTIDE MANAGEMENT
############################################

def loadIUPACcompatibilities(args):
	"""
		Generating a hash containing all compatibilities of all IUPAC RNA NUCLEOTIDES
	"""
	compatible = {}
	for nuc1 in args.IUPAC: # ITERATING OVER THE DIFFERENT GROUPS OF IUPAC CODE
		sn1 = list(args.IUPAC[nuc1])
		for nuc2 in args.IUPAC: # ITERATING OVER THE DIFFERENT GROUPS OF IUPAC CODE
			sn2 = list(args.IUPAC[nuc2])
			compatib = 0
			for c1 in sn1: # ITERATING OVER THE SINGLE NUCLEOTIDES WITHIN THE RESPECTIVE IUPAC CODE:
				for c2 in sn2: # ITERATING OVER THE SINGLE NUCLEOTIDES WITHIN THE RESPECTIVE IUPAC CODE:
					# CHECKING THEIR COMPATIBILITY
					if args.noGUBasePair == False:
						if (c1 == "A" and c2 == "U") or (c1 == "U" and c2 == "A") or (c1 == "C" and c2 == "G") or (c1 == "G" and c2 == "C") or (c1 == "G" and c2 == "U") or (c1 == "U" and c2 == "G"):
							compatib = 1
					else:
						if (c1 == "A" and c2 == "U") or (c1 == "U" and c2 == "A") or (c1 == "C" and c2 == "G") or (c1 == "G" and c2 == "C"):
							compatib = 1
			compatible[nuc1 + "_" + nuc2] = compatib # SAVING THE RESPECTIVE GROUP COMPATIBILITY, REVERSE SAVING IS NOT REQUIRED, SINCE ITERATING OVER ALL AGAINST ALL
	return compatible

def isCompatibleToSet(c1, c2, IUPAC_compatibles):
	"""
		Checks compatibility of c1 wihtin c2
	"""
	compatible = True
	for setmember in c2:
		if isCompatible(c1, setmember, IUPAC_compatibles) == False:
			return False
	return compatible


def isCompatible(c1, c2, IUPAC_compatibles):
	"""
		Checks compatibility between character c1 and c2
	"""
	if IUPAC_compatibles[c1.upper() + "_" + c2.upper()] == 1:
		return True
	else:
		return False

def compareLists(l1, l2):
	"""
		Compares reverse complement constraint lists.
	"""
	a = list(l1)
	b = list(l2)
	for i in a:
		if i not in b:
			return False
	return True

#######################################
# STRUCTURE AND DICTIONARY MANAGEMENT
#######################################

def getLP(BPSTACK):
	"""
		Retreives valid lonley base pairs from a base pair stack
	"""
	# getting single base pairs
	stack = {}
	LP = {}
	if type(BPSTACK[random.choice(BPSTACK.keys())]) == types.TupleType:
		for i in BPSTACK.keys():
			#if str(BPSTACK[i][1][0]) not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
			stack[i] = int(BPSTACK[i][1][0])
		## print i , BPSTACK[i][1][0]
	else:
		for i in BPSTACK.keys():
			#if str(BPSTACK[i]) not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
			stack[i] = BPSTACK[i]

	# removing redundant base pair indices
	for i in stack.keys():
		if i >= stack[i]:
			del stack[i]

	# actual checking for single lonley base pairs
	for i in stack.keys():
		if not (i-1 in stack and stack[i-1] == stack[i] + 1) and not (i+1 in stack and stack[i+1] == stack[i] - 1):
			LP[i] = stack[i]

	##actual removal of 2er lonley base pairs
	for i in stack.keys():
		if not (i-1 in stack and stack[i-1] == stack[i] + 1) and  (i+1 in stack and stack[i+1] == stack[i] - 1) and not (i+2 in stack and stack[i+2] == stack[i] - 2):
			LP[i] = stack[i]
			LP[i+1] = stack[i+1]
	return LP

def getBPStack(structure, sequence):
	"""
		Returns a dictionary of the corresponding basepairs of the structure s and the sequence constraint seq.
	"""
	tmp_stack = {"()":[], "{}":[], "[]":[], "<>":[]}
	BPstack = {}
	for i in range(len(structure)):

    # REGULAR SECONDARY STRUCTURE DETECTION
		if structure[i] in "(){}[]<>":

			no = 0
			### opening
			if structure[i] in "([{<":
				if structure[i] == "(":
					tmp_stack["()"].append((i, sequence[i]))
				elif structure[i] == "[":
					tmp_stack["[]"].append((i, sequence[i]))
				elif structure[i] == "{":
					tmp_stack["{}"].append((i, sequence[i]))
				elif structure[i] == "<":
					tmp_stack["<>"].append((i, sequence[i]))

			#closing
			elif structure[i] in ")]}>":
				if structure[i]  == ")":
					no, constr = tmp_stack["()"].pop()
				elif structure[i]  == "]":
					no, constr = tmp_stack["[]"].pop()
				elif structure[i]  == "}":
					no, constr = tmp_stack["{}"].pop()
				elif structure[i]  == ">":
					no, constr = tmp_stack["<>"].pop()
				BPstack[no] = (constr, (i, sequence[i]))
				BPstack[i] = (sequence[i] ,(no, constr))

		elif structure[i] == ".":
			BPstack[i] = (sequence[i], (i, sequence[i]))
		elif structure[i] in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz":
			BPstack[i] = (sequence[i], (i, sequence[i]))

	return (BPstack, getLP(BPstack))


def getbpStack(structure):
	"""
		Returns a dictionary of the corresponding basepairs of the structure s and the sequence constraint seq.
	"""
	tmp_stack = {"()":[], "{}":[], "[]":[], "<>":[]}
	bpstack = {}

	for i in range(len(structure)):
		if structure[i] in "(){}[]<>":

			no = 0
			### opening
			if structure[i] in "([{<":
				if structure[i] == "(":
					tmp_stack["()"].append(i)
				elif structure[i] == "[":
					tmp_stack["[]"].append(i)
				elif structure[i] == "{":
					tmp_stack["{}"].append(i)
				elif structure[i] == "<":
					tmp_stack["<>"].append(i)

			#closing
			elif structure[i] in ")]}>":
				if structure[i] == ")":
					no = tmp_stack["()"].pop()
				elif structure[i] == "]":
					no = tmp_stack["[]"].pop()
				elif structure[i] == "}":
					no = tmp_stack["{}"].pop()
				elif structure[i] == ">":
					no = tmp_stack["<>"].pop()
				bpstack[no] = i # save basepair in the format {opening base id (opening seq constr,(closing base id, closing seq constr))}
				bpstack[i] = no # save basepair in the format {closing base id (closing seq constr,(opening base id, opening seq constr))}

		elif structure[i] == ".": # no structural constaint given: produce entry, which references itself as a base pair partner....
			bpstack[i] = i

		elif structure[i] in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz":
			bpstack[i] = i

	return (bpstack, getLP(bpstack))

############################################
# TERRAIN GRAPH MANAGEMENT
############################################

def maprange( a, b, s):
	"""
		Mapping function
	"""
	(a1, a2), (b1, b2) = a, b
	return  b1 + ((s - a1) * (b2 - b1) / (a2 - a1))

def getConstraint(edge, args):
	"""
		Dependend on the situation in the constraint an the respective sequence section, setting weather a specific constraint can be given or not (for that sequence section)
	"""
	# terrain_element :: transition element / sequence section under dispute
	# id1 :: id of the position of the caharacter to which the transition is leading to
	# id2 :: id of the position of the character, which is listed in the BPinformation, it can be id1 as well, when no bp is present
	# val :: BPstack information of the specific position
	# constr1 :: constraining character of pos id1
	# constr2 :: constraining character of pos id2

	if args.modus == "MFE":
		# OLD VERSION
		## print edge
		id1 = int(edge.split(".")[0])
		targetNucleotide = edge.split(".")[1][-1:] # where the edge is leading to

		val = args.BPstack[id1] # check out the value of the destination character in the basepair/constraint stack
		constr1 = val[0] # getting the constraint character of position id1
		id2 = int(val[1][0]) # getting position id2
		constr2 = val[1][1] # getting the sequence constraint for position id2
		lowerCase = constr1.islower()
		if lowerCase:
			return 1
		else:
			c1 = set(args.IUPAC[constr1.upper()]) # getting all explicit symbols of c1
			c2 = set(args.IUPAC_reverseComplements[constr2.upper()]) # getting the reverse complement explicit symbols of c2
			if targetNucleotide in c1:
				if id1 == id2:
					return 1
				else:
					if targetNucleotide in c2:
						return 1
					else:
						return 0
			else:
				return 0

	if args.modus == "DP":

		x, y = edge # separate the edge into nodes
		## print edge
		id0, targetNucleotide0 = x.split(".") # retrieve node info
		id0 = int(id0)
		id1, targetNucleotide = y.split(".") # retrieve node info
		id1 = int(id1)

		check_nuc = ""
		check_id = -1

		if "XY" in targetNucleotide0:
			check_nuc = targetNucleotide
			check_id = id1


		if check_nuc == "": # transition is within the graph,indicating a base pair interaction, no consequtive nucleotide...
			## print id0, targetNucleotide0, id1, targetNucleotide,
			if isCompatible(targetNucleotide0, targetNucleotide, args.IUPAC_compatibles): # the specific base pair is possible, the edge will be further inspected. for constraints
				## print "is compatible"
				id0 += 1
				id1 += 1 # the interconnections are 1 based
				if id1 in args.interconnections: # nuc at pos id1 has some requested interaction

					constr0 = args.Cseq[id0 - 1]
					c0 = args.interconnections[id0][0]

					constr1 = args.Cseq[id1 - 1]
					c1 = args.interconnections[id1][0]

					if targetNucleotide0 in c0 and targetNucleotide in c1:
						for i in range(1,len(args.interconnections[id1])):
							c2 = "".join(set("".join([args.IUPAC_reverseComplements[i] for i in args.interconnections[id1][i][1]])))
							if targetNucleotide not in c2:
								return 0
						return 1
					else:
						return 0


				else: # case of no base pair
					constr1 = args.Cseq[id1-1]
					c1 = args.IUPAC[constr1.upper()]
					if targetNucleotide in c1:
						return 1
					else:
						return 0
			else:
				## print "is not compatible"
				return 0


		else: # emitting graph entry point
			# # print "Emitting", check_id, edge, check_id + 1
			if check_id + 1 in args.interconnections: # nucleotide is starting point of base pair interaction(s) IC are 1-based
				## print check_id + 1, args.interconnections[check_id + 1]
				targetNucleotides = set(args.interconnections[check_id + 1][0])

				if check_nuc in targetNucleotides:
					return 1
				else:
					return 0
			else: # unbound nucleotide situation
				return 1

def initTerrain(args):
	"""
		Initialization of the terrain graph  in equidistant mode.
		Vertices are modeled implicitly.
	"""
	NT = ["A","C","G","U"]
	NT2 = ["AA","AC","AG","AU","CA","CC","CG","CU","GA","GC","GG","GU","UA","UC","UG","UU"] # Allowed dinucleotides
	pathlength = 1
	pheromone = 1

	if args.modus == "MFE":
		e = {}
		for p in range(args.length):
			if p == 0:
				for i in NT:
					e["%s.%s"%(p,i)] = (pheromone, pathlength)
			elif p > 0:
				for n in NT2:
					e["%s.%s"%(p,n)] = (pheromone, pathlength)
		args.terrain = e

	elif args.modus == "DP":
		args.TerrainGraph = nx.DiGraph()
		args.TerrainGraph.add_node("HIVE")
		args.GenerateSequence = []
		graph_count = 0

		for graph in args.interconnection_graphs: # for all subgraphs
			start_node = random.choice(graph.nodes()) - 1 # randomly select a startpoint of the subgraph
			outpost = str(start_node)+".XY" # label of entry path to outpost subgraph
			args.TerrainGraph.add_edge("HIVE", outpost) # generate entry path to outpost

			for nt in NT: # generate all pathes from  outpost to first randomly chosen emitting battery of nodes

				target_node = "%s.%s"%(start_node,nt)
				args.TerrainGraph.add_edge(outpost, target_node, pheromone = pheromone, length = pathlength, status = "active")
			args.GenerateSequence.append((outpost, start_node))
			curr_node = start_node

			visited = []
			visit = [curr_node + 1]
			walk_order = []

			while len(visited) < graph.order():
				c_node = visit.pop()
				if c_node not in visited:
					visited.append(c_node)
				neighbor = ""
				try:
					neighbor = random.choice(list(set(graph.neighbors(c_node)) - set(visited)))
				except:
					pass
				if walk_order != []: # node is neighbor of an emmitting node which was visited before...
					for nt1 in NT:
						start_node = str(walk_order[-1]-1) + "." + nt1
						for nt2 in NT:
							target_node = str(c_node - 1) + "." + nt2
							args.TerrainGraph.add_edge(start_node, target_node, pheromone = pheromone, length = pathlength, status = "active")
					args.GenerateSequence.append((str(walk_order[-1]-1), str(c_node - 1)))

				if neighbor != "":
					walk_order.append(c_node)
					visit = [neighbor]
				else:
					if walk_order != []:
						go_back = walk_order.pop()
						visit = [go_back]

	## print args.GenerateSequence

def applyTerrainModification(args):
	"""
		Dependent on the input, this function modifies the terrain accordingly.
		It reflects the pheromone and path length adjustment.
		Directly affected edges and their implicit neighbor edges are removed
		from the graph.
	"""


	# determinde the respecitive complementarity chackings and gc length contribution for each edge
	actGC = {}
	for i in args.GC:
		v, s1, s2 = i
		for j in range(s1,s2 + 1):
			actGC[j] = v
	dels = []

	if args.modus == "MFE":
		dels = []
		for terrainelement in sorted(args.terrain):
			pheromone, pathlength = args.terrain[terrainelement]

			pheromone = getConstraint(terrainelement,args)
			pathlength = applyGCcontributionPathAdjustment(pathlength, actGC, terrainelement)

			if pheromone * pathlength == 0: dels.append(terrainelement)
			args.terrain[terrainelement] = (pheromone, pathlength,[])

		further_dels = {}
		for terrainelement in sorted(dels):
			pos, nucs = terrainelement.split(".")
			if int(pos) < len(args.Cseq)-1:
				to_nt = nucs[-1:]
				successor_pos = int(pos) + 1
				for i in ["A", "C", "G", "U"]:
					del_element = str(successor_pos) + "." + to_nt + i
					further_dels[del_element] = 1
			further_dels[terrainelement] = 1
		# deleting the inbound and outbound edges, which are forbidden
		for terrainelement in further_dels:
			del args.terrain[terrainelement]
		# allocate the appropriate children of edges
		for terrainelement in args.terrain:
			pheromone, pathlength, children = args.terrain[terrainelement]
			pos, nucs = terrainelement.split(".")
			if int(pos) < args.length:
				to_nt = nucs[-1:]
				successor_pos = int(pos) + 1
				for i in ["A", "C", "G", "U"]:
					if str(successor_pos) + "." + to_nt + i in args.terrain:
						children.append(str(successor_pos) + "." + to_nt + i)
			args.terrain[terrainelement] = (pheromone, pathlength,children)
		# ADDING THE START EDGES
		starts = []
		for i in ["A", "C", "G", "U"]:
			if str(0) + "." + i in args.terrain:
				starts.append(str(0) + "." + i)
		args.terrain["00.XY"] = (1, 1, starts)

		# CHECK IF THERE ARE EMPTY SUCCESSOR INFORMATION WITHIN THE SEQUENCE
		for terrainelement in args.terrain:
			pher, length, successors = args.terrain[terrainelement]
			id1 = terrainelement.split(".")[0]
			## print id1 ,len(args.Cseq), len(successors)
			if id1 < len(args.Cseq) and len(successors) == 0:
				args.error = "TerrainError: No successors detected where some should have appeared! %s %s" % (terrainelement, args.terrain[terrainelement])

	elif args.modus == "DP":

		for edge in args.TerrainGraph.edges():
			if "HIVE" not in edge:
				x, y = edge # split up of edge info into nodes

				pheromone = args.TerrainGraph[x][y]["pheromone"] # retrieve current pheromone value
				pathlength = args.TerrainGraph[x][y]["length"] # retrieve current pathlength value

				pheromone = getConstraint(edge,args) # get situation dependent pheromone value
				pathlength = applyGCcontributionPathAdjustment(pathlength, actGC, edge) # get situation dependent path length value

				if pheromone * pathlength == 0: # check if a factor is 0, therefore decide the edge to be not keepable for calculation
					dels.append(edge)
				else:
					args.TerrainGraph[x][y]["pheromone"] = pheromone
					args.TerrainGraph[x][y]["length"] = pathlength

		for edge in dels:
			x, y = edge
			args.TerrainGraph.remove_edge(x, y) # remove the unsuited edges from the Graph
			if len(args.TerrainGraph.neighbors(x)) + len(args.TerrainGraph.predecessors(x)) == 0:
				args.TerrainGraph.remove_node(x) # remove node x, if it has no in- and outgoing edges
			if len(args.TerrainGraph.neighbors(y)) + len(args.TerrainGraph.predecessors(y)) == 0:
				args.TerrainGraph.remove_node(y) # remove node y, if it has no in- and outgoing edges



def applyGCcontributionPathAdjustment(pathlength, actGC, edge):
	"""
		GC path length contribution calculation.
	"""
	# # print pathlength
	# # print actGC
	# # print edge
	# #exit(1)
	GCadjustment = 1.5
	minimum = 0.5
	upper = GCadjustment
	lower = minimum
	x = ""
	y = ""
	try:
		x, y = edge
	except:
		y = edge
	position , nts = y.split(".")
	nt = nts[-1:]

	if nt == "A" or nt == "U":
		pathlength = pathlength * maprange( (0, 1) , (lower, upper), actGC[int(position)])
	elif nt == "G" or nt == "C":
		pathlength = pathlength * maprange( (1, 0) , (lower, upper), actGC[int(position)])
	return pathlength

########################################
# SEQUENCE ASSEMBLY a.k.a. The ANTWALK
########################################

def checkSelection(sel):
	"""
		Check the selection for the presence of elements
	"""
	if len(sel[1]) == 0:
		# print "No legal nucleotides to fill in here. Please check your sequence constraint!"
		exit(1)

def pickNucleotide(args, selection):
	"""
		Selects a step within the terrain
	"""
	# # print selection
	# checkSelection(selection)

	if args.modus == "MFE":
		summe, tmp_steps = selection
		if len(tmp_steps) == 1:
			return tmp_steps[0][1] # returning the nucleotide of the only present step
		else:
			rand = random.random() # draw random number
			mainval = 0
			for choice in range(len(tmp_steps)):
				val, label = tmp_steps[choice]
				mainval += val/float(summe)
				if mainval > rand: # as soon, as the mainval gets larger than the random value the assignment is done
					return label

	elif args.modus == "DP":

		summe = 0
		selection_set = []
		for edge in selection:
				x, y = edge

				pheromone = args.TerrainGraph[x][y]["pheromone"]
				pathlength = args.TerrainGraph[x][y]["length"]

				value = ((float(pheromone * args.alpha)) + ((1/float(pathlength)) * args.beta))
				summe += value
				selection_set.append((value, edge))

		if len(selection_set) == 1:
			return selection_set[0][1] # returning the nucleotide of the only present step
		else:
			rand = random.random() # draw random number
			mainval = 0
			for choice in range(len(selection_set)):
				val, label = selection_set[choice]
				mainval += val/float(summe)
				if mainval > rand: # as soon, as the mainval gets larger than the random value the assignment is done
					return label

def getSequenceConstraint(args):
	"""
		Prerpare and renw the sequence constraint for the respective positions in Cseq
	"""
	args.sequenceConstraint = {}
	for i in range(args.length):
		if i + 1 in args.interconnections:
			args.sequenceConstraint[i] = args.interconnections[i + 1][0]
		else:
			args.sequenceConstraint[i] = args.IUPAC[args.Cseq[i]]

def getSequence(args):
	"""
		Performs a walk through the terrain and assembles a sequence, while respecting the structure constraint and IUPAC base complementarity
		of the base pairs GU, GC and AT
	"""

	nt = ["A","C","G","U"]
	path = ["_"] * args.length

	if args.modus == "MFE":

		prev_edge = "00.XY"
		sequence = ""

		while len(sequence) < args.length:
			coming_from = sequence[-1:]
			summe = 0
			steps = []
			i = len(sequence)
			allowed_nt = "ACGU"
			# base pair closing case check, with subsequent delivery of a reduced allowed nt set

			if i in args.BPstack:
				if i > args.BPstack[i][1][0]:
					jump =  args.BPstack[i][1][0]
					nuc_at_jump = sequence[jump]
					allowed_nt = args.IUPAC_reverseComplements[nuc_at_jump]

			for edge in args.terrain[prev_edge][-1]:

				if edge[-1:] in allowed_nt:
					pheromone, PL, children = args.terrain[edge]
					value = ((float(pheromone * args.alpha)) + ((1/float(PL)) * args.beta))
					summe += value
					steps.append((value, edge))

			if len(steps) > 0:
				prev_edge = pickNucleotide(args, (summe, steps) )
				pos, edge = prev_edge.split(".")
				sequence += edge[-1:]
				pos = int(pos)
				path[pos] = prev_edge
			else:
				# print "No legal nucleotides to fill in here. Please check your sequence constraint!"
				exit(1)
		return sequence, path

	elif args.modus == "DP":


		sequence = ["_"] * args.length

		getSequenceConstraint(args)

		for i, step in enumerate(args.GenerateSequence):

			source_node, target_node = step

			legal_target_node_nucleotides = list(args.sequenceConstraint[int(target_node)]) # get the legal list of nucleotides.due to sequence constraint
			if "XY" not in source_node:
				complements = list(args.IUPAC_reverseComplements[sequence[int(source_node)]])
				legal_target_node_nucleotides = [c for c in complements if c in legal_target_node_nucleotides]
				source_node = source_node + "." + sequence[int(source_node)]
			target_nodes = [str(target_node) + "." + n for n in legal_target_node_nucleotides]
			legal_edges = [ (source_node, s) for s in target_nodes]

			edge = pickNucleotide(args, legal_edges)
			position, emission = edge[1].split(".")

			sequence[int(position)] = emission
			path[int(position)] = edge

		return "".join(list(sequence)) , path

def getSequenceFromSelection(args, RNAfold, RNAfold_pattern):
	"""
		Returns the winning sequence from a selection of sequences...
	"""
	win_sequence = None
	for i in range(args.ants_per_selection): # for k ants do:
		# Generate Sequence
		sequence, path = getSequence(args)
		# # print sequence
		# # print path
		# Measure sequence features and transform them into singular distances
		ds, structure = getStructuralDistance(args, sequence, RNAfold, RNAfold_pattern)
		dGC = getGCDistance(sequence, args)
		dseq = getSequenceEditDistance(args.Cseq, sequence)
		# # print "ds",ds
		# # print "dgc",dGC, getGC(sequence)
		# # print "dseq", dseq
		# Calculate Distance Score
		D = ds + dGC + dseq
		# SELECT THE BEST-OUT-OF-k-SOLUTIONS according to distance score
		if i == 0: # Initial Case
			win_sequence = (sequence, structure, D, ds, dGC, dseq, path)
		else:
			if D < win_sequence[2]: # Challenge the Champion Case
				win_sequence = (sequence, structure, D, ds, dGC, dseq, path)
	# # print win_sequence[0], win_sequence[6], win_sequence[2]
	# exit(1)

	return win_sequence

################################
# STRUCTURE PREDICTION METHODS
################################

def getPKStructure(sequence, args):
	"""
		Initialization pKiss mfe pseudoknot prediction
	"""
	p2p = "pKiss_mfe"
	#p2p = "/usr/local/pkiss/2014-03-17/bin/pKiss_mfe"
	p = subprocess.Popen( ([p2p, '-T', str(args.temperature), '-s', args.strategy, sequence]),
				#shell = True,
				stdin = subprocess.PIPE,
				stdout = subprocess.PIPE,
				stderr = subprocess.PIPE,
				close_fds = True)

	# get linewise stdout output
	pKiss_output = p.stdout.read().split("\n");

	# init with empty sequence
	structure = "." * len(sequence)
	if (len(pKiss_output) > 1) :
		structure = "".join(pKiss_output[1].split(" ")[3])
	else :
		# print "DEBUG getPKStructure() : call stdout ="
		# print pKiss_output
		# print "DEBUG getPKStructure() : call stderr ="
                # print p.stderr.read()
		# print "DEBUG END (empty structure returned)"
		# close subprocess handle
		p.communicate()
		exit(-1)
	# close subprocess handle
	p.communicate()
	return structure

def getIPKnotStructure(sequence):
    """
        Initialization IPKnot pseudoknot prediction
    """
    f = str(uuid.uuid4())
    with open(f, 'w') as FASTA:
        FASTA.write(">tmp1\n")
        FASTA.write(sequence)
    p = subprocess.Popen( (['ipknot', f ]),
                #shell = True,
                stdin = subprocess.PIPE,
                stdout = subprocess.PIPE,
                stderr = subprocess.PIPE,
                close_fds = False)
    IPKnot_output = p.stdout.read().split("\n");
    # init with empty sequence
    structure = "." * len(sequence)
    if (len(IPKnot_output) > 1) :
        ## print IPKnot_output
        structure = IPKnot_output[2]
    else :
        # print "DEBUG getHKStructure() : call stdout ="
        # print IPKnot_output
        # print "DEBUG getHKStructure() : call stderr ="
        # print p.stderr.read()
        # print "DEBUG END (empty structure returned)"
        # close subprocess handle
        p.communicate()
        exit(-1)
    p.communicate()
    os.remove(f)
    return structure

def getHKStructure(sequence, args):
    """
        Initialization HotKnots pseudoknot prediction
    """
    curr_dir = os.getcwd()
    os.chdir(args.HotKnots_PATH)
    args = 'HotKnots -m CC -s ' + sequence
    p = subprocess.Popen( (args),
                shell = True,
                stdin = subprocess.PIPE,
                stdout = subprocess.PIPE,
                stderr = subprocess.PIPE,
                close_fds = False)

    HotKnots_output = p.stdout.read().split("\n");

    # init with empty sequence
    structure = "." * len(sequence)
    if (len(HotKnots_output) > 1) :
        structure = "".join(HotKnots_output[2].replace(" ", "#").replace("##","#").split("#")[1].split("\t")[0])
    else :
        # print "DEBUG getHKStructure() : call stdout ="
        # print HotKnots_output
        # print "DEBUG getHKStructure() : call stderr ="
        # print p.stderr.read()
        # print "DEBUG END (empty structure returned)"
        # close subprocess handle
        p.communicate()
        exit(-1)
    p.communicate()
    os.chdir(curr_dir)
    return structure


def init_RNAfold(version, temperature, paramFile = ""):
	"""
		Initialization RNAfold listener
	"""
	p2p = ""
	t = "-T " + str(temperature)
	P = ""
	if paramFile != "":
		P = "-P " + paramFile
	p2p = "RNAfold"
	p = subprocess.Popen( ([p2p, '--noPS', '-d 2', t, P]),
				#shell = True,
				stdin = subprocess.PIPE,
				stdout = subprocess.PIPE,
				stderr = subprocess.PIPE,
				close_fds = True)
	return p

def consult_RNAfold(seq, p):
	"""
		Consults RNAfold listener
	"""
	p.stdin.write(seq+'\n')
	out = ""
	for i in range(2):
		out += p.stdout.readline()
	return out


def getRNAfoldStructure(struct2, process1):
	"""
		Retrieves folded structure of a RNAfold call
	"""

	RNAfold_pattern = re.compile('.+\n([.()]+)\s.+')
	RNAfold_match = RNAfold_pattern.match(consult_RNAfold(struct2, process1))
	current_structure = ""
	return RNAfold_match.group(1)



###########################################
# DISTANCE MEASURES AND RELATED FUNCTIONS
###########################################

def getInducingSequencePositions(args):
	"""
		Delimiting the degree of structure inducement by the supplied sequence constraint.
		0 : no sequence induced structure constraint
		1 : "ACGT" induce structure (explicit nucleotide structure inducement level)
		2 : "MWKSYR" and "ACGT" (explicit and double instances)
		3 : "BDHV" , "MWKSYR" and "ACGT" (explicit, double, and triple instances)
	"""
	setOfNucleotides = "" # resembling the "0"-case
	if args.level == 1:
		setOfNucleotides = "ACGU"
	elif args.level == 2:
		setOfNucleotides = "ACGUMWKSYR"
	elif args.level == 3:
		setOfNucleotides = "ACGUMWKSYRBDHV"

	tmpSeq = ""
	listset = setOfNucleotides
	for pos in args.Cseq:
		if pos not in listset:
			tmpSeq += "N"
		else:
			tmpSeq += pos

	return setOfNucleotides, tmpSeq

def getBPDifferenceDistance(stack1, stack2):
	"""
		Based on the not identical amount of base pairs within both structure stacks
	"""
	d = 0
	for i in stack1.keys():
		# check base pairs in stack 1
		if i < stack1[i] and stack1[i] != stack2[i]:
			d += 1
		# check base pairs in stack 2
	for i in stack2.keys():
		if i < stack2[i] and stack1[i] != stack2[i]:
			d += 1
	return d

def getMaxTargetDeviation(targetValue):
    """
        Retruns the maximally obtainalble deviation of a taregtValue in correspondance with [0,1]
    """

    if targetValue >= 0.5:
        return targetValue
    else:
        return 1 - targetValue


def getStructuralDistance(args, sequence, RNAfold, RNAfold_pattern):
	"""
		Calculator for Structural Distance
	"""
	# fold the current solution's sequence to obtain the structure
	current_structure = ""
	### Selection of specific folding mechanism
	if args.modus == "MFE":
		if args.pseudoknots:
			if args.pkprogram == "pKiss":
				current_structure = getPKStructure(sequence, args)
			elif args.pkprogram == "HotKnots":
				current_structure = getHKStructure(sequence, args)
			elif args.pkprogram == "IPKnot":
				current_structure = getIPKnotStructure(sequence)
		else:
			RNAfold_match = RNAfold_pattern.match(consult_RNAfold(sequence, RNAfold))
			current_structure = RNAfold_match.group(1)



		# generate the current structure's base-pair stack
		bp = getbpStack(current_structure)[0]

		# deriving a working copy of the target structure# add case-dependend structural constraints in case of lonley basepairs formation
		tmp_target_structure_bp = getbpStack(args.Cstr)[0]

		### LONELY BASE PAIR MANAGEMENT
		# add case-dependend structural constraints in case of lonley basepairs formation
		if args.noLBPmanagement:
			for lp in args.LP:
				if bp[lp] == args.LP[lp]: # if the base pair is within the current solution structure, re-add the basepair into the constraint structure.
					tmp_target_structure_bp[lp] = args.LP[lp]
					tmp_target_structure_bp[args.LP[lp]] = lp
		###
		### "ABCDEFGHIJKLMNOPQRSTUVWXYZ" -> HARD CONSTRAINT
		### "abcdefghijklmnopqrstuvwxyz" -> SOFT CONSTRAINT
		###

		# FUZZY STRUCTURE CONSTRAINT MANAGEMENT - HARD CONSTRAINT
		# check for all allowed hard implicit constraint block declarators
		dsoftCstr = 0
		for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz":
			occurances = []
			for m in re.finditer(c, args.Cstr): # search for a declarator in the requested structure
				occurances.append(m.start()) # save the corresponding index
			# transform declarator into single stranded request
			for i in occurances:
				tmp_target_structure_bp[i] = i
			# infer a base pair within the block declarated positions, if the current structure provides it.
			fuzzy_block_penalty = 1
			for i in occurances:
				for j in occurances:
					if i < j:
						if bp[i] == j:
							fuzzy_block_penalty = 0 # if a base pair is present within a current fuzzy_block, no penalty is risen
							tmp_target_structure_bp[i] = bp[i]
							tmp_target_structure_bp[bp[i]] = i
			if len(occurances) > 0:
				if c.isupper():
					dsoftCstr += fuzzy_block_penalty

		# INDUCES STRUCTURE MANAGEMENT
		# Checking Cseq influence and it's induced basepairs
		IUPACinducers, tmp_Cseq = getInducingSequencePositions(args)
		if len(args.Cseq.strip("N")) > 0:
		## print "Processing Cseq influence"
			# Iterate over all positions within the Base Pair stack
			for i in args.BPstack: # Check for each base index i
				if i < bp[i]: # if the current index is samller that the affiliated in the basepair stack of the current solution
					bp_j = bp[i] # Actual j index of the current solution
					BP_j = args.BPstack[i][1][0] # j index of the requested structure
					if (i != bp_j and i == BP_j and args.BPstack[i][0] in IUPACinducers ): # if i pairs with some other base in the current structure, and i is requested single stranded and the Sequence constraint is allowed to induce...
						if (args.BPstack[bp_j][1][0] == bp_j and args.BPstack[bp_j][0] in IUPACinducers):# If position j is requested singlestranded and position j nucleotide can induce base pairs
							#if isCompatible(bp[i][0], bp[i][1][1], IUPAC_compatibles): # If both nucleotides, i and j are actually compatible
							tmp_target_structure_bp[i] = bp[i]
							tmp_target_structure_bp[bp_j] = i

		dsreg = getBPDifferenceDistance(tmp_target_structure_bp, bp)

		# CHECK FOR ALL DETERMINED LONELY BASE PAIRS (i<j), if they are formed
		failLP = 0
		if args.noLBPmanagement:
			for lp in args.LP:
				if bp[lp] != args.LP[lp]:
					isComp = isCompatible(sequence[lp],sequence[args.LP[lp]], args.IUPAC_compatibles)
					isStru = isStructureCompatible(lp, args.LP[lp] ,bp)
					if not ( isStru and isStru ): # check if the bases at the specific positions are compatible and check if the
						# basepair can be formed according to pseudoknot free restriction. If one fails, a penalty distance is raised for that base pair
						failLP += 1

		dsLP = float(failLP)
		return (dsreg + dsLP + dsoftCstr) /float(len(tmp_target_structure_bp)) * 100, current_structure


	elif args.modus == "DP":
		max_struct_deviation = 0
		L = len(sequence)
		DP = {}

		# Calculating the Dotplot for the unconstrained case
		DP["UB"] = getBPPM(sequence)

		# Checking if some specific Cstr constraint was inputed, in case: calculate a constrained Dotplot
		if args.Cstr is not None:
			# making a copy of Cstr hackingstyle, such that the constraint is not overwritten by the getBPPM() function
			tmp_struct = (args.Cstr + "x")[:-1]
			# calculating the Dotplot for the constrained case
			DP["B"] =  getBPPM(sequence, tmp_struct)



		# Differential Accessibility Structural Feature
		## print "DiffAccessibility:"
		ddsf_access = 0
		if args.diffaccess:
			for i in args.diffaccess:
				## print i
				diff1 = 0
				diff2 = 0
				for c in i[0]:
					tmp_c = {c:c}
					# calculating the first deviation
					access_1 = getAccessibility(tmp_c, L, DP[i[3]])
					diff1 += abs(access_1 - i[4])
					# calculating the second deviation
					access_2 = getAccessibility(tmp_c, L, DP[i[1]])
					diff2 += abs(access_2 - i[2])

					max_struct_deviation += getMaxTargetDeviation(i[2])
					max_struct_deviation += getMaxTargetDeviation(i[4])
				diff1 /= float(len(i[0]))
				diff2 /= float(len(i[0]))
				ddsf_access += diff1 + diff2


		# Differential Accuracy Structural Feature
		ddsf_diff_accur = 0
		if args.diffaccur:
			for i in args.diffaccur:
				diff1 = 0
				diff2 = 0
				for c in i[0]:
					tmp_c = {c:i[0][c]}
					# calculating the first deviation
					accur_1 = getAccuracy(tmp_c, DP[i[3]])
					diff1 += abs(accur_1 - i[4])
					# calculating the second deviation
					accur_2 = getAccuracy(tmp_c, DP[i[1]])
					diff2 += abs(accur_2 - i[2])

					max_struct_deviation += getMaxTargetDeviation(i[4])
					max_struct_deviation += getMaxTargetDeviation(i[2])
				diff1 /= float(len(i[0]))
				diff2 /= float(len(i[0]))
				ddsf_diff_accur += diff1 + diff2



		# Accessibility structural feature
		dsf_access = 0
		if args.access:
			for i in args.access:

				diff1 = 0
				for c in i[0]:
					tmp_c = {c:c}
					# calculating the deviation
					access_1 = abs(getAccessibility(tmp_c, L, DP[i[1]]))
					diff1 += abs(access_1 - i[2])

					max_struct_deviation += getMaxTargetDeviation(i[2])
				diff1 /= float(len(i[0]))
				dsf_access += diff1

		# Accuracy structural feature
		dsf_accur = 0
		if args.accur:
			for i in args.accur:
				# # print "i", i
				diff1 = 0
				for c in i[0]:
					tmp_c = {c:i[0][c]}
					# # print "mpt_c", tmp_c,
					# calculating the deviation
					accu_1 = getAccuracy(tmp_c, DP[i[1]])
					diff1 += abs(accu_1 - i[2])
					# # print "accu", accu_1,
					# # print "Diff1", diff1,
					max_struct_deviation += getMaxTargetDeviation(i[2])
					# # print "MSD", max_struct_deviation,
				diff1 /= float(len(i[0]))
				# # print "ndiff", diff1
				dsf_accur += diff1

		# Fuzzy Constraint Distance
		# # print "DistancE"

		dsf_fuzzy = 0
		# # print dsf_fuzzy
		if args.fuzzy:
			for i in args.fuzzy:
				# # print i
				diff1 = 0
				for c in i[0]:
					a,b = c
					tmp_c = {a:b}

					fuzzy_1 = getAccuracy(tmp_c, DP[i[1]])
					# fuzzy_1 /= len(i[0])**2
					diff1 += abs(fuzzy_1 - i[2])
				## print diff1
				diff1 /= float(len(i[0]))
				max_struct_deviation += getMaxTargetDeviation(i[2])
				dsf_fuzzy += diff1


		ddsf_fuzzy = 0
		if args.difffuzzy:
			for i in args.difffuzzy:
				## print i
				diff1 = 0
				diff2 = 0
				# fuzzy_1 = 0
				# fuzzy_2 = 0
				for c in i[0]:
					a,b = c
					tmp_c = {a:b}

					fuzzy_1 = getAccuracy(tmp_c, DP[i[1]])
					fuzzy_2 = getAccuracy(tmp_c, DP[i[3]])

				# fuzzy_1 /= len(i[0])**2
				# fuzzy_2 /= len(i[0])**2
					diff1 += abs(fuzzy_1 - i[2])
					diff2 += abs(fuzzy_2 - i[4])
				## print diff_1
				diff1 /= float(len(i[0]))
				diff2 /= float(len(i[0]))
				max_struct_deviation += getMaxTargetDeviation(i[2])
				max_struct_deviation += getMaxTargetDeviation(i[4])
				ddsf_fuzzy += diff1 + diff2






		ddsf = ddsf_diff_accur + ddsf_access + ddsf_fuzzy
		dsf = dsf_accur + dsf_access + dsf_fuzzy
		# # print "DSF", dsf, "MSD", max_struct_deviation
		d = ((ddsf + dsf) / max_struct_deviation) * 100
		# # print d

		# # print "dfaccur", ddsf_diff_accur
		# # print "dfaccuess", ddsf_access
		# # print "dffuzzy", ddsf_fuzzy
		# # print "accur", dsf_accur
		# # print "access", dsf_access
		# # print "fuzzy", dsf_fuzzy
		# # print "dev", max_struct_deviation
		# # print "d", d
		return d , DP


def getGC(sequence):
	"""
		Calculate GC content of a sequence
	"""
	GC = 0
	for nt in sequence:
		if nt == "G" or nt == "C":
			GC = GC + 1
	GC = GC/float(len(sequence))
	return GC

def getGCDistance( sequence, args):
	"""
		Calculate the pseudo GC content distance
	"""
	D = 0
	for tGC in args.GC:
		v, s1, s2 = tGC # (tGC, start, stop)
		tmp_seq = sequence[s1 : s2 + 1]
		L = len(tmp_seq)
		gc = getGC(tmp_seq)
		nt_coeff = L * v
		pc_nt = (1/float(L))*100
		d = gc - v
		d = d * 100
		f = math.floor(nt_coeff)
		c = math.ceil(nt_coeff)
		if d < 0:
			d = d + (abs(nt_coeff - f) * pc_nt)
		elif d > 0:
			d = d - (abs(nt_coeff - c) * pc_nt)
		elif d == 0:
			pass

		d = abs(round(d, 7))
		D += d
	return D


def getSequenceEditDistance(Cseq, sequence):
	"""
		Calculate sequence edit distance of a solution to the constraint
	"""
	IUPAC = {"A":"A", "C":"C", "G":"G", "U":"U", "R":"AG", "Y":"CU", "S":"GC", "W":"AU","K":"GU", "M":"AC", "B":"CGU", "D":"AGU", "H":"ACU", "V":"ACG", "N":"ACGU"}
	edit = 0
	for i in range(len(Cseq)):
		if sequence[i] not in IUPAC[Cseq[i].upper()]:
			edit += 1
	return edit/float(len(sequence))


#########################
# TERRAIN GRAPH EDITING
#########################

def evaporate(args):
	"""
		Evaporate the terrain's pheromone trails
	"""
	if args.modus == "MFE":
		c = 1
		for key in args.terrain:
			p, l, c = args.terrain[key]
			p *= (1 - args.ER)
			args.terrain[key] = (p, l, c)

	elif args.modus == "DP":
		for edge in [e for e in args.TerrainGraph.edges() if "HIVE" not in e[0]]:
			x, y = edge
			args.TerrainGraph[x][y]["pheromone"] *= (1 - args.ER)


def trailBlaze(sequence, path, current_structure, ds, dgc, dseq, args):
	"""
		Pheromone Update function accorinding to the quality of the solution
	"""
	## print "trail blaze"
	bs = updateValue(ds, args.Cstrweight, args.omega)
	bGC = updateValue(dgc, args.Cgcweight, args.omega)
	bSeq = updateValue(dseq, args.Cseqweight, args.omega)
	d = bs + bGC + bSeq

	#transitions = getTransitions(sequence)
	transitions = path
	## print args
	if args.modus == "MFE":

		bpstack, LP = getbpStack(current_structure)
		for trans in range(len(transitions)): # for each transition in the path
			id1 = int(transitions[trans].split(".")[0])
			tar_id2 = int(args.BPstack[id1][1][0]) # getting requested  position id2
			curr_id2 = int(bpstack[id1]) # getting the current situation
			multiplicator = 0

			if tar_id2 == curr_id2 and id1 != tar_id2 and id1 != curr_id2: # case of a base pair, having both brackets on the correct position
				multiplicator = 1
			elif tar_id2 == curr_id2 and id1 == tar_id2 and id1 == curr_id2: # case of a single stranded base in both structures
				multiplicator = 1
			p, l, c = args.terrain[transitions[trans]] # getting the pheromone and the length value of the single path transition
			p +=  d * multiplicator
			args.terrain[transitions[trans]] = (p, l, c) # updating the values wihtin the terrain's

	elif args.modus == "DP":
		"""
			If a Fuzzy structure constraint is present, bonify the according currently participating edges accordingly
		"""

		if args.fuzzy:
			for elements in args.fuzzy:
				base_pairs = elements[0]
				indices = set([])
				for bp in base_pairs:
					a,b = bp
					indices = indices | set([a])
					indices = indices | set([b])

				system = elements[1]
				value = elements[2]

				d = 0
				for bp in base_pairs:
					a, b = bp
					tmp_stack = {a:b}
					d += getAccuracy(tmp_stack, current_structure[system])
				d /= float(len(base_pairs))
				difference = abs(d - value)

				if difference <= args.trailblazer:
					edges_to_bonify = []
					for edge_step in args.GenerateSequence:

						a,b = edge_step

						if int(b) in indices:
							indx = args.GenerateSequence.index(edge_step)
							## print  edge_step , indx
							edge_instance = path[int(b)]
							aa, bb = edge_instance
							args.TerrainGraph[aa][bb]["pheromone"] +=  d * (1-difference)
		if args.difffuzzy:
			for elements in args.difffuzzy:
				base_pairs = elements[0]
				indices = set([])
				for bp in base_pairs:
					a,b = bp
					indices = indices | set([a])
					indices = indices | set([b])

				system1 = elements[1]
				value1 = elements[2]
				system2 = elements[3]
				value2 = elements[4]

				d1 = 0
				d2 = 0
				for bp in base_pairs:
					a, b = bp
					tmp_stack = {a:b}
					d1 += getAccuracy(tmp_stack, current_structure[system1])
					d2 += getAccuracy(tmp_stack, current_structure[system2])
				d1 /= float(len(base_pairs))
				d2 /= float(len(base_pairs))
				difference1 = abs(d1 - value)
				difference2 = abs(d2 - value)
				difference = (difference1 + difference2) / float(2)

				if difference <= args.trailblazer:
					edges_to_bonify = []
					for edge_step in args.GenerateSequence:

						a,b = edge_step

						if int(b) in indices:
							indx = args.GenerateSequence.index(edge_step)
							## print  edge_step , indx
							edge_instance = path[int(b)]
							aa, bb = edge_instance
							args.TerrainGraph[aa][bb]["pheromone"] +=  d * (1-difference)

		"""
			If a complete interaction collection is performing well, the whole interconnection is bonified by average quality of the group
		"""
		## print "GS", args.GenerateSequence


		for i, edge in enumerate(args.GenerateSequence):
			## print "\n"
			# # print "walked", i, edge

			# extract info about subgraph to investigate
			## print "HellO"
			batch = []
			if "XY" in edge[0]: # starting point to investigate a subgraph and create a corresponding batch
				batch.append(edge)
				j = i
				if j + 1 < len(args.GenerateSequence)-1: # increment for following edges
						j += 1
				else:
					pass
				# add edges to the batch until the next "XY" label
				# # print args.GenerateSequence
				# # print args.GenerateSequence[j]
				# # print args.GenerateSequence[j][0]
				while "XY" not in args.GenerateSequence[j][0]:
					batch.append(args.GenerateSequence[j])
					if j + 1 < len(args.GenerateSequence)-1:
						j += 1
					else:
						break
			## print "SooO"
			# extract respective nodes which are involved in the subgraph
			batch_individuals = set([])
			## print "Batch", batch
			for element in batch:
				element0, element1 = element
				for e in [element0, element1]:
					if "XY" not in str(e):
						batch_individuals = batch_individuals | set([str(e)])
			batch_individuals = sorted(list(batch_individuals))

			# check for each individual of the batch, how it performs in the dotplots and retreive
			# the corresponding deviation from the target value
			values = []
			individual_values = {}
			## print "batch individual ", batch_individuals
			for h in batch_individuals:
				h = int(h)
				ii = h + 1
				# # print "ii", ii
				for DP in args.PosFeatures[ii]:
					# # print DP
					if len(args.PosFeatures[ii][DP]) > 0:
						for k, feature in enumerate(args.PosFeatures[ii][DP]):
							feature_type, jj, value = feature
							## print "feature", feature
							if feature_type == "Accu":
								tmp_stack = {ii:jj}
								## print "stack", tmp_stack
								val = getAccuracy(tmp_stack, current_structure[DP])
								result_value = abs(val - value)
								values.append(result_value)

							elif feature_type == "Accs":
								tmp_stack = {ii:jj}
								## print "stack", tmp_stack
								val = getAccessibility(tmp_stack, len(args.Cseq), current_structure[DP])
								result_value = abs(val - value)
								values.append(result_value)

			if len(values) > 0: # calculate an average value of the just vitnessed results
				avrg_val = sum(values)/float(len(values))
			else: # or assume the worst case, maximal deviation
				avrg_val = 1

			# bonify the members of the batch if the average is below a certain threshold
			if  avrg_val <= args.trailblazer:
				for element in batch_individuals:  # retreive corresponding edge, which led to the respective nucleotide
					sequence_position = int(element)
					from_nt, to_nt = path[sequence_position]
					# average quality threshold to trigger bonification with average value
					args.TerrainGraph[from_nt][to_nt]["pheromone"] +=  d * (1-avrg_val) # bonify with average batch value


def getTransitions(p):
	"""
		Retreive transitions of a specific path/sequence
	"""
	transitions = []
	for pos in range(len(p)):
		if pos == 0:
			transitions.append(str(pos) + "." + p[pos])

		else:
			insert = p[pos-1] + p[pos]
			transitions.append(str(pos) + "." + insert)

	return transitions

def transformTransitions(transitions):

	T = {}
	for t in transitions:
		index, letters = t.split(".")
		T[index] = letters
	return T

def updateValue(distance, correction_term, omega):
	"""
		Retrieves a distance dependend pheromone value
	"""
	if correction_term == 0:
		return 0
	else:
		if distance == 0:
			return omega * correction_term
		else:
			return (1/float(distance)) * correction_term

def updateTerrain(sequence, path, current_structure, ds, dgc, dseq, args):
	"""
		General updating function
	"""
	evaporate(args)
	trailBlaze(sequence, path,  current_structure, ds, dgc, dseq, args)



#######################
# CONVERGENCE MEASURE
#######################

def inConvergenceCorridor(d_struct, d_gc, d_seq, BS_d_struct, BS_d_gc, BS_d_seq):
	"""
		Check if a solutions qualities are within the convergence corridor
	"""

	# # print d_struct, d_gc, d_seq, BS_d_struct, BS_d_gc, BS_d_seq
	struct_var = (BS_d_struct + BS_d_struct/float(100) * 5)
	gc_var = (BS_d_gc + BS_d_gc/float(100) * 5)
	seq_var = (BS_d_seq + BS_d_seq/float(100) * 5)

	# # print struct_var, gc_var, seq_var

	if d_struct <= struct_var and d_gc <= gc_var and d_seq <= seq_var:
		# # print "Korridor True"
		return True
	else:
		# # print "Korridor False"
		return False



##############################
# TARGET GC VALUE MANAGEMENT
##############################

def getGCSamplingValue(GC, tGCmax, tGCvar):
	"""
		Returns a suitable GC value, dependend on the user input: Either returning the single GC value,
		which the user entered, or a smpled GC value
		from a designated distribution in it's interavals
	"""
	returnval = 0
	if tGCmax == -1.0 and tGCvar == -1.0: # regular plain tGC value as requested
		return GC
	elif tGCmax != -1.0 and tGCvar == -1.0: # uniform distribution tGC value sampling
		if GC < tGCmax:
			tmp_GC = tGCmax
			tGCmax = GC
			GC = tmp_GC
		while returnval <= 0:
			returnval = float(np.random.uniform(low=GC, high=tGCmax, size=1))
		return returnval
	elif tGCmax == -1.0 and tGCvar != -1.0: # normal distribution tGC value sampling
		while returnval <= 0:
			returnval = float(np.random.normal(GC, tGCvar, 1))
		return returnval

def setGC(args):
	"""
		checks the tGC situation and maps it to GC arguments
	"""
	args.GC = []
	if len(args.tGC) == 1:
		args.GC = [(getGCSamplingValue(args.tGC[0][0], args.tGCmax, args.tGCvar), args.tGC[0][1], args.tGC[0][2])]
	else:

		args.GC = args.tGC



def exe():
	"""
		MAIN EXECUTABLE WHICH PARSES THE INPUT COMMAND LINE
	"""

	argument_parser = argparse.ArgumentParser(
	fromfile_prefix_chars='@',
	description = """

	#########################################################################
	#       antaRNA - ant assembled RNA                                     #
	#       -> Ant Colony Optimized RNA Sequence Design                     #
	#       ------------------------------------------------------------    #
	#       Robert Kleinkauf (c) 2015                                       #
	#       Bioinformatics, Albert-Ludwigs University Freiburg, Germany     #
	#########################################################################

	- For antaRNA only the VIENNNA RNA Package must be installed on your linux system.
	  antaRNA will only check, if the executables of RNAfold  of the ViennaRNA package can be found. If those programs are
	  not installed correctly, no output will be generated, an also no warning will be prompted.
	  So the binary path of the Vienna Tools must be set up correctly in your system's PATH variable in order to run antaRNA correctly!
	- If you want to use the pseudoknot functionality, pKiss_mfe of the RNAshapes studio OR HotKnots OR IPKnot must be installed and callable as standalone in order to execute antaRNA.

    - antaRNA was only tested under Linux.

    - For questions and remarks please feel free to contact us at http://www.bioinf.uni-freiburg.de/

	""",

	epilog = """
	Example calls:
		python antaRNA_vXY.py -Cstr "...(((...)))..." -tGC 0.5 -n 2 -v
		python antaRNA_vXY.py -Cstr ".........aaa(((...)))aaa........." -tGC 0.5 -n 10 --output_file /path/to/antaRNA_TESTRUN -v
		python antaRNA_vXY.py -Cstr "BBBBB....AAA(((...)))AAA....BBBBB" -Cseq "NNNNANNNNNCNNNNNNNNNNNGNNNNNNUNNN" --tGC 0.5 -n 10

	#########################################################################
	#       --- Hail to the Queen!!! All power to the swarm!!! ---          #
	#########################################################################
		""",
	formatter_class=RawTextHelpFormatter
	)

	argument_parser.convert_arg_line_to_args = convert_arg_line_to_args

	subparsers = argument_parser.add_subparsers(help='\'MFE\' (minimum free energy) or \'DP\' (dotplot) mode selection', dest="subparser_name")

	### MFE PARSER
	MFE_parser = subparsers.add_parser('MFE', help='MFE mode: compute an RNA sequence according to the mfe model of a structure. Required for the pseudoknot variant.')
	MFE_parser.add_argument("-Cstr", "--Cstr",
								help="Structure constraint using RNA dotbracket notation with fuzzy block constraint. \n(TYPE: %(type)s)\n\n",
								type=str,
								required=True)

	pk = MFE_parser.add_argument_group('Pseudoknot Variables', 'Use in order to enable pseudoknot calculation. pKiss_mfe needs to be installed.')
	pk.add_argument("-p", "--pseudoknots",
								help = "Switch to pseudoknot based prediction using pKiss. Check the pseudoknot parameter usage!!!\n\n",
								action="store_true")

	pk.add_argument("-pkPar", "--pkparameter",
								help = "Enable optimized parameters for the usage of pseudo knots (Further parameter input ignored).\n\n",
								action="store_true")
	pk.add_argument("-pkP", "--pkprogram",
								help = "Select a pseudoknot prediction program.\nIf HotKnots is used, please specify the bin folder of Hotknots with absolute path using HK_PATH argument.\n(DEFAULT: %(default)s, TYPE: %(type)s, Choice: [pKiss|HotKnots|IPKnot])\n\n",
								type=str,
								default="pKiss")
	pk.add_argument("-HKPATH", "--HotKnots_PATH",
								help = "Set HotKnots absolute path, like /path/to/HotKnots/bin.\nIf HotKnots is used, please specify the bin folder of Hotknots with absolute path using HK_PATH argument.\n(DEFAULT: %(default)s, TYPE: %(type)s\n\n",
								type=str,
								default="")

	pk.add_argument("--strategy",
								help = "Defining the pKiss folding strategy.\n(DEFAULT: %(default)s, TYPE: %(type)s)\n\n",
								type=str,
								default="A")

	### DP PARSER
	DP_parser = subparsers.add_parser('DP', help='DP mode: compute an RNA sequence according to the dotplot(s) model.')
	DP_parser.add_argument("-Cstr", "--Cstr",
								help="Structure constraint using RNA dotbracket notation. If specified, this structure will be used to constrain a folding hypothesis to produce a ligand bound model of the dotplot.\n(TYPE: %(type)s)\n\n",
								type=str,
								default = None)
	DP_parser.add_argument("--accuracy",
								help="Define an accuracy evaluation block.\n\n",
								type=AccuracyFeature,
								default=None,
								action='append'
								)
	DP_parser.add_argument("--accessibility",
								help="Define an accessibility evaluation block.\n\n",
								type=AccessibilityFeature,
								default=None,
								action='append'
								)
	DP_parser.add_argument("--diff-accuracy",
								#help="Define an differential accuracy evaluation block.\n\n",
								type=DiffAccuracyFeature,
								default=None,
								action='append'
								)
	DP_parser.add_argument("--diff-accessibility",
								help="Define an differential accessibility evaluation block.\n\n",
								type=DiffAccessibilityFeature,
								default=None,
								action='append'
								)

	DP_parser.add_argument("--fuzzyConstraint",
								help="Define fuzzy structure constraint wihtin DP mode\n\n",
								type=FuzzyConstraintFeature,
								default=None,
								action='append'
								)
	DP_parser.add_argument("--diff-fuzzyConstraint",
								help="Define a differential fuzzy structure constraint wihtin DP mode\n\n",
								type=DiffFuzzyConstraintFeature,
								default=None,
								action='append'
								)

	DP_parser.add_argument("--trailblaze_threshold",
								help="Define the threshold whic need to be passed in order to bonify certain elements in the terrain graph.\n\n",
								type=float,
								default=0.2
								)
	### General Variables available in both modes
	constraints = argument_parser.add_argument_group('Constraint Variables', 'Use to define an RNA constraint system.')


	constraints.add_argument("-Cseq", "--Cseq",
								help="Sequence constraint using RNA nucleotide alphabet {A,C,G,U} and wild-card \"N\". \n(TYPE: %(type)s)\n\n",
								type=str,
								default = None)

	constraints.add_argument("-l", "--level",
								help="Sets the level of allowed influence of sequence constraint on the structure constraint [0:no influence; 3:extensive influence].\n(TYPE: %(type)s)\n\n",
								type=int,
								default = 1)

	constraints.add_argument("-tGC", "--tGC",
								help="Objective target GC content in [0,1].\n(TYPE: %(type)s)\n\n",
								type=parseGC,
								required=True,
								action = 'append')

	constraints.add_argument("-tGCmax", "--tGCmax",
								help = "Provides a maximum tGC value [0,1] for the case of uniform distribution sampling. The regular tGC value serves as minimum value.\n(DEFAULT: %(default)s, TYPE: %(type)s)\n\n",
								type=float,
								default=-1.0)

	constraints.add_argument("-tGCvar", "--tGCvar",
								help = "Provides a tGC variance (sigma square) for the case of normal distribution sampling. The regular tGC value serves as expectation value (mu).\n(DEFAULT: %(default)s, TYPE: %(type)s)\n\n",
								type=float,
								default=-1.0)

	constraints.add_argument("-T", "--temperature",
								help = "Provides a temperature for the folding algorithms.\n(DEFAULT: %(default)s, TYPE: %(type)s)\n\n",
								type=float,
								default=37.0)

	constraints.add_argument("-P", "--paramFile",
								help = "Changes the energy parameterfile of RNAfold. If using this explicitly, please provide a suitable energy file delivered by RNAfold. \n(DEFAULT: %(default)s, TYPE: %(type)s)\n\n",
								type=str,
								default="")

	constraints.add_argument("-noGU", "--noGUBasePair",
								help="Forbid GU base pairs. \n\n",
								action="store_true")

	constraints.add_argument("-noLBP", "--noLBPmanagement",
								help="Disallowing antaRNA lonely base pair-management. \n\n",
								action="store_false",
								default =True)

	output = argument_parser.add_argument_group('Output Variables', 'Tweak form and verbosity of output.')
	output.add_argument("-n", "--noOfColonies",
								help="Number of sequences which shall be produced. \n(TYPE: %(type)s)\n\n",
								type=int,
								default=1)

	output.add_argument("-of","--output_file",
								help="Provide a path and an output file, e.g. \"/path/to/the/target_file\". \n(DEFAULT: %(default)s, TYPE: %(type)s)\n\n",
								type=str,
								default="STDOUT")

	output.add_argument("-rPY","--py",
								help="Switch on PYTHON compatible behavior. \n(DEFAULT: %(default)s)\n\n",
								action="store_true")

	output.add_argument("--name",
								help="Defines a name which is used in the sequence output. \n(DEFAULT: %(default)s, TYPE: %(type)s)\n\n",
								type=str,
								default="antaRNA")

	output.add_argument("-v", "--verbose",
								help="Displayes intermediate output.\n\n",
								action="store_true")

	output.add_argument("-ov", "--output_verbose",
								help="Prints additional features and stats to the headers of the produced sequences. Also adds the structure of the sequence.\n\n",
								action="store_true")
	output.add_argument("--plot",
								help="# print Terrain Nodes and edges files.\n\n",
								action="store_true")



	aco = argument_parser.add_argument_group('Ant Colony Variables', 'Alter the behavior of the ant colony optimization.')
	aco.add_argument("-s", "--seed",
								help = "Provides a seed value for the used pseudo random number generator.\n(DEFAULT: %(default)s, TYPE: %(type)s)\n\n",
								type=str,
								default=None)



	aco.add_argument("-ip", "--improve_procedure",
								help = "Select the improving method.  h=hierarchical, s=score_based.\n(DEFAULT: %(default)s, TYPE: %(type)s)\n\n",
								type=str,
								default="s")

	aco.add_argument("-r", "--Resets",
								help = "Amount of maximal terrain resets, until the best solution is retuned as solution.\n(DEFAULT: %(default)s, TYPE: %(type)s)\n\n",
								type=int,
								default=5)

	aco.add_argument("-aps", "--ants_per_selection",
								help = "best out of k ants.\n(DEFAULT: %(default)s, TYPE: %(type)s)\n\n",
								type=int,
								default=10)

	aco.add_argument("-CC", "--ConvergenceCount",
								help = "Delimits the convergence count criterion for a reset.\n(DEFAULT: %(default)s, TYPE: %(type)s)\n\n",
								type=int,
								default=130)

	aco.add_argument("-aTC", "--antsTerConv",
								help = "Delimits the amount of internal ants for termination convergence criterion for a reset.\n(DEFAULT: %(default)s, TYPE: %(type)s)\n\n",
								type=int,
								default=50)
	aco.add_argument("--alpha",
								help="Sets alpha, probability weight for terrain pheromone influence. [0,1] \n(DEFAULT: %(default)s, TYPE: %(type)s)\n\n",
								type=float,
								default=1.0)

	aco.add_argument("--beta",
								help="Sets beta, probability weight for terrain path influence. [0,1]\n(DEFAULT: %(default)s, TYPE: %(type)s)\n\n",
								type=float,
								default=1.0)
	aco.add_argument("--omega",
								help="Sets the value, which is used in the mimiced 1/x evaluation function in order to set a crossing point on the y-axis.",
								type=float,
								default=2.23)

	aco.add_argument("-er", "--ER",
								help="Pheromone evaporation rate. \n(DEFAULT: %(default)s, TYPE: %(type)s)\n\n",
								type=float,
								default=0.2)

	aco.add_argument("-Cstrw", "--Cstrweight",
								help="Structure constraint quality weighting factor. [0,1]\n(DEFAULT: %(default)s, TYPE: %(type)s)\n\n",
								type=float,
								default=0.5)

	aco.add_argument("-Cgcw", "--Cgcweight",
								help="GC content constraint quality weighting factor. [0,1]\n(DEFAULT: %(default)s, TYPE: %(type)s)\n\n",
								type=float,
								default=5.0)

	aco.add_argument("-Cseqw", "--Cseqweight",
								help="Sequence constraint quality weighting factor. [0,1]\n(DEFAULT: %(default)s, TYPE: %(type)s)\n\n\n",
								type=float,
								default=1.0)

	aco.add_argument("-t", "--time",
								help="Limiting runtime [seconds]\n(DEFAULT: %(default)s, TYPE: %(type)s)\n\n\n",
								type=int,
								default=600)

	argparse_arguments = argument_parser.parse_args()

	print(argparse_arguments)


	hill = AntHill()
	hill.params.readArgParseArguments(argparse_arguments)
	hill.params.py = False
	hill.params.check()





	if hill.params.error == "0":
		hill.swarm()
	else:
		# print hill.params.error






#########################
### CLASSES
#########################
		"""
		___________________________/\/\______/\/\____/\/\__/\/\____/\/\____/\/\___
		_/\/\/\______/\/\/\/\____/\/\/\/\/\__/\/\____/\/\__________/\/\____/\/\___
		_____/\/\____/\/\__/\/\____/\/\______/\/\/\/\/\/\__/\/\____/\/\____/\/\___
		_/\/\/\/\____/\/\__/\/\____/\/\______/\/\____/\/\__/\/\____/\/\____/\/\___
		_/\/\/\/\/\__/\/\__/\/\____/\/\/\____/\/\____/\/\__/\/\/\__/\/\/\__/\/\/\_
		__________________________________________________________________________

		"""
class AntHill:
	"""
		antaRNA AntHill. Can perform varoius actions! :)
	"""

	def __init__(self):

		self.params = Variables()
		self.tmp_sequence = ""
		self.tmp_path = []
		self.tmp_terrain = ""
		self.tmp_structure = ""
		self.tmp_stats = []
		self.tmp_result = []
		self.result = []



	###################################################
	#  PRINCIPAL EXECUTION ROUTINES OF THE ACO METHOD
	###################################################

	def swarm(self):
		"""
			Execution function of a single ant colony finding one solution sequence
		"""
		RNAfold = init_RNAfold(213, self.params.temperature, self.params.paramFile)
		RNAfold_pattern = re.compile('.+\n([.()]+)\s.+')

		for n in range(self.params.noOfColonies):

			start_time = time.time()

			setGC(self.params)
			initTerrain(self.params)
			applyTerrainModification(self.params)
			# with open(os.getcwd() + "/antaRNA_edges", 'w') as TARGET:
			# 	TARGET.write("Source;Target;Weight\n")
			# 	for edges in self.params.TerrainGraph.edges():
			# 		n1, n2 = edges
			# 		pheromone = 1
			# 		try:
			# 			pheromone = self.params.TerrainGraph[n1][n2]["pheromone"]
			# 		except:
			# 			pass
			# 		TARGET.write(n1 + ";" + n2 + ";" + str(pheromone)+"\n")
			# with open(os.getcwd() + "/antaRNA_nodes", 'w') as TARGET:
			# 	TARGET.write("Nodes;Id;Label\n")
			# 	for node in self.params.TerrainGraph.nodes():
			# 		TARGET.write("\""+node + "\";\"" + node + "\";\"" + node + "\"\n")
			# exit(1)
			global_ant_count = 0
			global_best_ants = 0
			criterion = False
			met = True
			ant_no = 1
			seq = ""

			counter = 0

			quality_log = []
			quality_log_c = []
			dstruct_log = []
			dGC_log = []

			convergence_counter = 0

			resets = 0
			sequence = ""
			current_structure = ""
			path = []

			Dscore = 100000
			ds = 10000
			dGC = 10000
			dseq = 10000
			best_solution = (sequence, current_structure, Dscore, ds, dGC, dseq, path)
			best_solution_local = (sequence, current_structure, Dscore, ds, dGC, dseq, path)

			best_solution_since = 0

			winning_terrain = nx.Graph()

			# IN CASE OF LP-MANAGEMENT
			if self.params.modus == "MFE":
				if self.params.noLBPmanagement:
					if len(self.params.LP) > 0 :
						for lp in self.params.LP:
							self.params.Cstr = substr(lp + 1, self.params.Cstr, ".")
							self.params.Cstr = substr(self.params.LP[lp] + 1, self.params.Cstr, ".")

			used_time = getUsedTime(start_time)
			while criterion != met and used_time < self.params.time:
				iteration_start = time.time()
				global_ant_count += 1
				global_best_ants += 1

				sequence_info = getSequenceFromSelection(self.params, RNAfold, RNAfold_pattern)

				distance_structural_prev = ds
				distance_GC_prev = dGC
				distance_seq_prev = dseq
				sequence, current_structure, Dscore , ds, dGC, dseq , path = sequence_info
				curr_solution = (sequence, current_structure, Dscore, ds, dGC, dseq, path)
				# BEST SOLUTION PICKING
				if self.params.improve_procedure == "h": # hierarchical check
					# for the global best solution
					if ds < best_solution[3] or (ds == best_solution[3] and dGC < best_solution[4]):
						best_solution = curr_solution
						winning_terrain = self.params.TerrainGraph.copy()
						ant_no = 1
					# for the local (reset) best solution
					if ds < best_solution_local[3] or (ds == best_solution_local[3] and dGC < best_solution_local[4]):
						best_solution_local = curr_solution

				elif self.params.improve_procedure == "s": #score based check
					# store best global solution

					if Dscore < best_solution[2]:
						best_solution = curr_solution
						if self.params.modus == "MFE":
							winning_terrain = self.params.terrain
						elif self.params.modus == "DP":
							winning_terrain = self.params.TerrainGraph.copy()
						ant_no = 1
					# store best local solution for this reset
					if Dscore < best_solution_local[2]:
						best_solution_local = curr_solution



				if self.params.verbose:
					# print "SCORE " + str(Dscore) + " Resets " + str(resets) + " #Ant " + str(global_ant_count) + " out of " + str(self.params.ants_per_selection)  + " cc " + str(convergence_counter)
					if self.params.modus == "MFE":
						pass
						# print self.params.Cstr, " <- target struct"
					# # print best_solution[0] , " <- BS since ", str(best_solution_since), "Size of Terrrain:", len(self.params.terrain)
					# print best_solution[0] , " <- BS since ", str(best_solution_since), "Size of Terrrain:",
					if self.params.modus == "MFE":
						pass
						# print len(self.params.terrain)
					elif self.params.modus == "DP":
						pass
						# print len(self.params.TerrainGraph)
					if self.params.modus == "MFE":
						pass
						# print best_solution[1] , " <- BS"
					# print " Dscore " + str(best_solution[2]) + " ds " + str(best_solution[3]) + " dGC " + str(best_solution[4]) + " dseq " + str(best_solution[5])+ " LP " + str(len(self.params.LP)) + " <- best solution stats"
					if self.params.modus == "MFE":
						pass
						# print current_structure, " <- CS"
					# print sequence,
					# print " <- CS", "Dscore", str(Dscore), "ds", ds, "dGC", dGC, "GC", getGC(sequence)*100, "Dseq", dseq
					# dstruct_log.append((global_ant_count, Dscore, best_solution[2]))
				#### UPDATING THE TERRAIN ACCORDING TO THE QUALITY OF THE CURRENT BESTO-OUT-OF-k SOLUTION
				## print [self.params.TerrainGraph[e[0]][e[1]]["pheromone"]for e in self.params.TerrainGraph.edges() if "HIVE" not in e[0]]
				updateTerrain(sequence, path, current_structure, ds, dGC, dseq, self.params)
				## print [self.params.TerrainGraph[e[0]][e[1]]["pheromone"]for e in self.params.TerrainGraph.edges() if "HIVE" not in e[0]]
				# exit(1)


				if self.params.verbose: # print "Used time for one iteration", time.time() - iteration_start
					pass


				# CONVERGENCE AND TERMINATION CRITERION MANAGEMENT
				if inConvergenceCorridor(curr_solution[3], curr_solution[4], curr_solution[5], best_solution_local[3], best_solution_local[4], best_solution_local[5]):
					convergence_counter += 1





				if distance_structural_prev == ds and distance_GC_prev == dGC and distance_seq_prev == dseq:
					convergence_counter += 1

				if best_solution[3] == self.params.objective_to_target_distance:
					if best_solution[4] == 0.0:
						if best_solution[5] == 0.0:
							break
					ant_no = ant_no + 1
					convergence_counter -= 1
				else:
					ant_no = 1


				quality_log.append(best_solution[2])
				quality_log_c.append(len(quality_log_c))
				if len(quality_log) > 50:
					quality_log.pop(0)
					quality_log_c.pop()
					p1 = np.polyfit(quality_log_c,quality_log,1)
					slope = p1[0]
					## print "Slope", slope
					if slope > -0.05:
						## print "Convergence setting"
						convergence_counter = self.params.ConvergenceCount + 1

				if ant_no == self.params.antsTerConv or resets >= self.params.Resets or global_ant_count >= 100000 or best_solution_since == 5:
					# print ant_no, self.params.antsTerConv, resets , self.params.Resets
					break

				# RESET
				if ant_no < self.params.antsTerConv and convergence_counter >= self.params.ConvergenceCount:
					quality_log = []
					quality_log_c = []
					initTerrain(self.params)
					applyTerrainModification(self.params)
					criterion = False
					met = True
					ant_no = 1
					sequence = ""
					current_structure = ""
					path = []
					counter = 0
					Dscore = 100000
					ds = 10000
					dGC = 10000
					dseq = 10000
					best_solution_local = (sequence, current_structure, Dscore, ds, dGC, dseq, path)

					convergence_counter = 0

					if resets == 0:
						sentinel_solution = best_solution
						best_solution_since += 1
					else:
						if best_solution[3] < sentinel_solution[3]:
							sentinel_solution = best_solution
							best_solution_since = 0
						else:
							best_solution_since += 1

					resets += 1

				used_time = getUsedTime(start_time)

			duration  = used_time
			self.tmp_sequence = best_solution[0]
			self.tmp_path = best_solution[6]
			if self.params.modus == "MFE":
				self.tmp_structure = best_solution[1]
			self.tmp_stats.append("Ants:" + str(global_ant_count))
			self.tmp_stats.append("Resets:" + str(resets) + "/" + str(self.params.Resets))
			self.tmp_stats.append("AntsTC:" + str(self.params.antsTerConv))
			self.tmp_stats.append("CC:" + str(self.params.ConvergenceCount))
			self.tmp_stats.append("IP:" + str(self.params.improve_procedure))
			self.tmp_stats.append("BSS:" + str(best_solution_since))
			if self.params.LP:
				self.tmp_stats.append("LP:" + str(len(self.params.LP)))
			self.tmp_stats.append("ds:" + str(best_solution[3]))
			#self.tmp_stats.append("ds:" + str(getStructuralDistance(self.params, sequence, RNAfold, RNAfold_pattern)))
			self.tmp_stats.append("dGC:" + str(best_solution[4]))
			self.tmp_stats.append("GC:" + str(getGC(self.tmp_sequence)*100))

			self.tmp_stats.append("dseq:" + str(best_solution[5]))
			self.tmp_stats.append("L:" + str(len(self.tmp_sequence)))
			self.tmp_stats.append("Time:" + str(duration))

			self.retrieveResult(n)

			if self.params.plot == True:

				# if self.params.modus == "MFE":


				# 	with open(os.getcwd() + "/antaRNA_nodes", 'w') as TARGET:
				# 		TARGET.write("Nodes;Id;Label\n")
				# 		TARGET.write("\"XY.00\";\"XY.00\";\"XY.00\"\n")
				# 		for node in  winning_terrain.keys(): # nodes
				# 			pos, nucs = node.split(".")
				# 			node = pos + "." + nucs[-1:]
				# 			TARGET.write("\""+node + "\";\"" + node + "\";\"" + node + "\"\n")

				# 	with open(os.getcwd() + "/antaRNA_edges", 'w') as TARGET:
				# 	 	TARGET.write("Source;Target;Weight\n")
				# 		for node in winning_terrain.keys():

				# 			pos, nucs = node.split(".")

				# 			if len(nucs) == 1:
				# 				phero = winning_terrain[node][0]
				# 				TARGET.write("XY.0;" + target + ";" + str(phero)+"\n")

				# 			elif len(nucs) == 2:
				# 				pos = int(pos)
				# 				source = str(pos-1) + "." + nucs[0]
				# 				target = str(pos) + "." + nucs[1]
				# 				phero = winning_terrain[node][0]
				# 				TARGET.write(source + ";" + target + ";" + str(phero)+"\n")
					#
					# 	for edges in winning_terrain.edges():
					# 		n1, n2 = edges
					# 		pheromone = 1
					# 		try:
					# 			pheromone = winning_terrain[n1][n2]["pheromone"]
					# 		except:
					# 			pass
					# 		TARGET.write(n1 + ";" + n2 + ";" + str(pheromone)+"\n")

					# with open(os.getcwd() + "/antaRNA_nodes", 'w') as TARGET:
					# 	TARGET.write("Nodes;Id;Label\n")
					# 	for node in winning_terrain.nodes():
					# 		TARGET.write("\""+node + "\";\"" + node + "\";\"" + node + "\"\n")

				if self.params.modus == "DP":
					## print self.params.GenerateSequence
					## print self.tmp_path
					with open(os.getcwd() + "/antaRNA_edges", 'w') as TARGET:
						TARGET.write("Source;Target;Weight\n")
						for edges in winning_terrain.edges():
							n1, n2 = edges
							pheromone = 1
							try:
								pheromone = winning_terrain[n1][n2]["pheromone"]
							except:
								pass
							TARGET.write(n1 + ";" + n2 + ";" + str(pheromone)+"\n")
					with open(os.getcwd() + "/antaRNA_nodes", 'w') as TARGET:
						TARGET.write("Nodes;Id;Label\n")
						for node in winning_terrain.nodes():
							TARGET.write("\""+node + "\";\"" + node + "\";\"" + node + "\"\n")


		# CLOSING THE PIPES TO THE PROGRAMS
		if (RNAfold is not None) :
			RNAfold.communicate()







	def retrieveResult(self,col):
		"""
			Collect the results which have been produced by swarm function.
		"""


		# Post-Processing the output of a ant colony procedure

		self.tmp_result = [">" + self.params.name + "#" + str(col)]
		if self.params.output_verbose:

			GC_out = ""
			for i in self.params.GC:
				v, s1, s2 = i
				GC_out += str(s1) + "-" + str(s2) + ">" + str(v) + ";"
			GC_out = GC_out[:-1]

			if self.params.Cstr:
				self.tmp_result.append("Cstr:" + self.params.Cstr)
			self.tmp_result.append("Cseq:" + self.params.Cseq)
			self.tmp_result.append("Alpha:" + str(self.params.alpha))
			self.tmp_result.append("Beta:" + str(self.params.beta))
			self.tmp_result.append("tGC:" + str(GC_out))
			self.tmp_result.append("ER:" + str(self.params.ER))
			self.tmp_result.append("Struct_CT:" + str(self.params.Cstrweight))
			self.tmp_result.append("GC_CT:" + str(self.params.Cgcweight))
			self.tmp_result.append("Seq_CT:" + str(self.params.Cseqweight))
			self.tmp_result.append("UsedProgram:" + self.params.usedProgram)
			self.tmp_result.append("Modus:" + self.params.modus)
			self.tmp_result.extend(self.tmp_stats)

			self.tmp_result.append("Rseq:" + self.tmp_sequence)
			if self.params.modus == "MFE":
				self.tmp_result.append("Rstr:" + self.tmp_structure)

		else:
			self.tmp_result.append("Rseq:" + self.tmp_sequence)


		if self.params.py == False:
			if self.params.print_to_STDOUT:
				if len(self.tmp_result) > 2:
					struct = self.tmp_result.pop()
					seq = self.tmp_result.pop()
					# print "|".join(self.tmp_result)
					# print seq
					# print struct
				else:
					for i, entry in enumerate(self.tmp_result):
						pass
						# print entry.replace("Rseq:", "")
			else:
				if col == 0:
					print2file(self.params.output_file, "\n".join(self.tmp_result), 'w')
				else:
					print2file(self.params.output_file, "\n".join(self.tmp_result), 'a')
		else:
			## print self.tmp_result
			self.result.append(tuple(self.tmp_result))


		if self.params.print_to_STDOUT == False:
			os.chdir(curr_dir)


class Variables:
	"""
	_/\/\____/\/\__________________________/\/\________________/\/\________/\/\___________________________
	_/\/\____/\/\__/\/\/\______/\/\__/\/\__________/\/\/\______/\/\________/\/\______/\/\/\______/\/\/\/\_
	_/\/\____/\/\______/\/\____/\/\/\/\____/\/\________/\/\____/\/\/\/\____/\/\____/\/\/\/\/\__/\/\/\/\___
	___/\/\/\/\____/\/\/\/\____/\/\________/\/\____/\/\/\/\____/\/\__/\/\__/\/\____/\/\______________/\/\_
	_____/\/\______/\/\/\/\/\__/\/\________/\/\/\__/\/\/\/\/\__/\/\/\/\____/\/\/\____/\/\/\/\__/\/\/\/\___
	______________________________________________________________________________________________________

	"""
	"""
		antaRNA Variables management.
	"""
	def __init__(self):
		self.modus = "MFE"
		self.Cstr = ""
		self.accuracy = []
		self.accessibility = []
		self.diff_accuracy = []
		self.diff_accessibility = []
		self.fuzzyCstr = []
		self.diff_fuzzyCstr = []
		self.length = 0
		self.Cseq = None
		self.tGC = []
		self.level = 1
		self.tGCmax = -1.0
		self.tGCvar = -1.0
		self.temperature = 37.0
		self.paramFile = ""
		self.noGUBasePair = False
		self.noLBPmanagement = True
		self.pseudoknots = False
		self.usedProgram = "RNAfold"
		self.pkprogram = "pKiss"
		self.pkparameter = False
		self.HotKnots_PATH = ""
		self.strategy = "A"
		self.noOfColonies = 1
		self.output_file = "STDOUT"
		self.py = True
		self.name="antarna"
		self.verbose = False
		self.output_verbose = False
		self.seed ="none"
		self.improve_procedure = "s"
		self.Resets = 5
		self.ants_per_selection = 10
		self.ConvergenceCount = 130
		self.antsTerConv = 50
		self.alpha = 1.0
		self.beta = 1.0
		self.ER = 0.2
		self.Cstrweight = 0.5
		self.Cgcweight = 5.0
		self.Cseqweight = 1.0
		self.omega = 2.23
		self.time = 600
		self.plot = False
		self.error = "0"
		self.trailblazer = 0.2


	def readArgParseArguments(self, args):
		"""
			Read the argparse arguments
		"""
		## print args
		self.modus = args.subparser_name
		if self.modus == "MFE":
			self.Cstr = args.Cstr
			self.level = args.level
			self.tGCmax = args.tGCmax
			self.tGCvar = args.tGCvar
			self.pseudoknots = args.pseudoknots
			self.pkprogram = args.pkprogram
			self.pkparameter = args.pkparameter
			self.HotKnots_PATH = args.HotKnots_PATH
			self.strategy = args.strategy
		elif self.modus == "DP":
			self.Cstr = args.Cstr
			self.accuracy = args.accuracy
			self.accessibility = args.accessibility
			self.diff_accuracy = args.diff_accuracy
			self.diff_accessibility = args.diff_accessibility
			self.fuzzyCstr = args.fuzzyConstraint
			self.diff_fuzzyCstr = args.diff_fuzzyConstraint
			self.trailblazer = args.trailblaze_threshold

		self.Cseq = args.Cseq
		self.tGC = args.tGC

		self.temperature = args.temperature
		self.paramFile = args.paramFile
		self.noGUBasePair = args.noGUBasePair
		self.noLBPmanagement = args.noLBPmanagement

		self.noOfColonies = args.noOfColonies
		self.output_file = args.output_file
		self.py = args.py
		self.name = args.name
		self.verbose = args .verbose
		self.output_verbose = args.output_verbose
		self.seed = args.seed
		self.improve_procedure = args.improve_procedure
		self.Resets = args.Resets
		self.ants_per_selection = args.ants_per_selection
		self.ConvergenceCount = args.ConvergenceCount
		self.antsTerConv = args.antsTerConv
		self.alpha = args.alpha
		self.beta = args.beta
		self.ER = args.ER
		self.Cstrweight = args.Cstrweight
		self.Cgcweight = args.Cgcweight
		self.Cseqweight = args.Cseqweight
		self.omega = args.omega
		self.time = args.time
		self.plot = args.plot

	####################################################
	# STRUCTURE AND SEQUENCE INTEGRITY CHECK FUNCTIONS
	####################################################

	def checkSequenceConstraint(self):
		"""
			Checks the Sequence constraint for illegal nucleotide characters
		"""
		for c in self.Cseq:
			if c not in "ACGURYSWKMBDHVNacgu":
				self.error = "Used Cseq -> %s <- is not a valid sequence constraint" % (c)

	def checkSimilarLength(self):
		"""
			Compares sequence and structure constraint length
		"""
		if len(self.Cstr) != len(self.Cseq):
			self.error =  "Constraint have different lengths: Cstr: " + str(len(self.Cstr)) + ",Cseq: " + str(len(self.Cseq))

	def isStructure(self):
		"""
			Checks if the structure constraint only contains "(", ")", and "." and legal fuzzy structure constraint characters.
		"""
		for a in range(0,len(self.Cstr)):
			if self.Cstr[a] not in  ".()[]{}<>":
				if self.Cstr[a] not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz":
					self.error = "Specified structure is not a valid structure. Illegal characters detected: " + self.Cstr[a]

	def isBalanced(self):
		"""
			Check if the structure s is of a balanced nature
		"""
		for bracket in ["()", "[]", "{}", "<>"]:
			counter = 0
			for a in range(len(self.Cstr)):
				if self.Cstr[a] in bracket[0]:
					counter += 1
				elif self.Cstr[a] in bracket[1]:
					counter -= 1
			if counter != 0:
				self.error = "Structure is not balanced."

	def fulfillsHairpinRule(self):
		"""
			CHECKING FOR THE 3 nt LOOP INTERSPACE
				for all kind of basepairs, even wihtin the pdeudoknots
		"""
		for bracket in ["()", "[]", "{}", "<>"]:
			last_opening_char = 0
			tmp_check = 0
			for a in range(len(self.Cstr)):
				if self.Cstr[a] == bracket[0]:
					last_opening_char = a
					tmp_check = 1
				elif self.Cstr[a] == bracket[1] and tmp_check == 1:
					tmp_check = 0
					if a - last_opening_char < 4:
						self.error = "Hairpin loopsize rule violation"

	def checkConstaintCompatibility(self):
		"""
			Checks if the constraints are compatible to each other
		"""
		self.BPstack, self.LP = getBPStack(self.Cstr, self.Cseq)
		## print self.BPstack
		for id1 in self.BPstack:  # key = (constraint , (pos, constraint)))
			constr1 = self.BPstack[id1][0]
			id2 = self.BPstack[id1][1][0]
			constr2 = self.BPstack[id1][1][1]
			if id1 != id2 and not isCompatible(constr1, constr2, self.IUPAC_compatibles):
				self.error = "Contraint Compatibility Issue: Nucleotide constraint " + str(constr1) + " at position " + str(id1) + " is not compatible with nucleotide constraint " + str(constr2) + " at position " + str(id2)

	def transform(self):
		"""
			Transforms "U" to "T" for the processing is done on DNA alphabet
		"""
		S = ""
		for s in self.Cseq:
			if s == "T":
				S += "U"
			elif s == "t":
				S += "u"
			else:
				S += s
		self.Cseq = S


	def check_FuzzyCstr(self):
		"""
			Check the made Fuzzy-Cstr Features
		"""
		for i in self.fuzzyCstr:
			## print i
			s1, s2, s3 = i
			if s2 != "UB" and s2 != "B":
				self.error = "FuzzyConstraintError: Wrongly defined Accessiblity", s1, s2, s3, "->", s2
			if not isfloat(s3):
				self.error = "FuzzyConstraintError: Wrongly defined Accessibility", s1, s2, s3, "->" , s3
			if len(s1) != self.length:
				self.error = "FuzzyConstraintError Constraint Length", len(s1), "is unequeal to Cstr length", self.length, "!"


	def check_Diff_FuzzyCstr(self):
		"""
			Check the made differential Fuzzy-Cstr Features
		"""
		for i in self.diff_fuzzyCstr:
			s1, s2,s3, s4, s5 = i
			if not isfloat(s3):
				self.error = "DiffFuzzyConstraintError: Wrongly defined DiffAccessibility", s1, s2,s3, s4, s5, "->" , s3
			if not isfloat(s5):
				self.error = "DiffFuzzyConstraintError: Wrongly defined DiffAccessibility", s1, s2,s3, s4, s5, "->" , s5
			if s3 < 0 or s3 > 1:
				self.error = "DiffFuzzyConstraintError: Value",s3, "must remain in range [0,1]."
			if s5 < 0 or s5 > 1:
				self.error = "DiffFuzzyConstraintError: Value",s3, "must remain in range [0,1]."
			if s2 != "UB" and s2 != "B":
				self.error = "DiffFuzzyConstraintError: Value",s2, "must be selected from {\"UB\", \"B\"}."
			if s4 != "UB" and s4 != "B":
				self.error = "DiffFuzzyConstraintError: Value",s4, "must be selected from {\"UB\", \"B\"}."
			if (s4 == "B" and s2 == "B") or (s4 == "UB" and s2 == "UB"):
				self.error = "DiffFuzzyConstraintError: The constraint systems mus be different!."
			if len(s1) != self.length:
				self.error = "DiffFuzzyConstraintError: Constraint Length", len(s1), "is unequeal to Cstr length", self.length, "!"


	def check_Accessibilities(self):
		"""
			Check made accessibility statements
		"""
		for i in self.accessibility:
			## print i
			s1, s2, s3 = i
			if s2 != "UB" and s2 != "B":
				self.error = "AccessibilityError: Wrongly defined Accessiblity", s1, s2, s3, "->", s2
			if not isfloat(s3):
				self.error = "AccessibilityError: Wrongly defined Accessibility", s1, s2, s3, "->" , s3
			if len(s1) != self.length:
				self.error = "AccessibilityError Constraint Length", len(s1), "is unequeal to Cstr length", self.length, "!"


	def check_Diff_Accessibilities(self):
		"""
			Check made differential accessibility statements
		"""
		for i in self.diff_accessibility:
			s1, s2,s3, s4, s5 = i
			if not isfloat(s3):
				self.error = "DiffAccessibilityError: Wrongly defined DiffAccessibility", s1, s2,s3, s4, s5, "->" , s3
			if not isfloat(s5):
				self.error = "DiffAccessibilityError: Wrongly defined DiffAccessibility", s1, s2,s3, s4, s5, "->" , s5
			if s3 < 0 or s3 > 1:
				self.error = "DiffAccessibilityError: Value",s3, "must remain in range [0,1]."
			if s5 < 0 or s5 > 1:
				self.error = "DiffAccessibilityError: Value",s3, "must remain in range [0,1]."
			if s2 != "UB" and s2 != "B":
				self.error = "DiffAccessibilityError: Value",s2, "must be selected from {\"UB\", \"B\"}."
			if s4 != "UB" and s4 != "B":
				self.error = "DiffAccessibilityError: Value",s4, "must be selected from {\"UB\", \"B\"}."
			if (s4 == "B" and s2 == "B") or (s4 == "UB" and s2 == "UB"):
				self.error = "DiffAccessibilityError: The constraint systems mus be different!."
			if len(s1) != self.length:
				self.error = "DiffAccessibilityError: Constraint Length", len(s1), "is unequeal to Cstr length", self.length, "!"

	def check_Accuracies(self):
		"""
			Check made accuraciy statements
		"""
		## print self.accuracy
		for i in self.accuracy:
			## print i
			s1, s2, s3 = i
			if s2 != "UB" and s2 != "B":
				self.error = "AccuracysError:: Wrongly defined Accuracy", s1, s2, s3, "->", s2
			if not isfloat(s3):
				self.error = "AccuracysError::Wrongly defined Accuracy", s1, s2, s3, "->" , s3
			if len(s1) != self.length:
				self.error = "AccuracyError: Constraint Length", len(s1), "is unequeal to Cstr length", self.length, "!"

	def check_Diff_Accuracies(self):
		"""
			Check made differential accuracy statements
		"""
		for i in self.diff_accuracy:
			s1, s2,s3, s4, s5 = i
			if not isfloat(s3):
				self.error = "DiffAccuracyError: Wrongly defined DiffAccuracy", s1, s2,s3, s4, s5, "->" , s3
			if not isfloat(s5):
				self.error = "DiffAccuracyError: Wrongly defined DiffAccuracy", s1, s2,s3, s4, s5, "->" , s5
			if s3 < 0 or s3 > 1:
				self.error = "DiffAccuracyError: Value",s3, "must remain in range [0,1]."
			if s5 < 0 or s5 > 1:
				self.error = "DiffAccuracyError: Value",s3, "must remain in range [0,1]."
			if s2 != "UB" and s2 != "B":
				self.error = "DiffAccuracyError: Value",s2, "must be selected from {\"UB\", \"B\"}."
			if s4 != "UB" and s4 != "B":
				self.error = "DiffAccuracyError: Value",s4, "must be selected from {\"UB\", \"B\"}."
			if (s4 == "B" and s2 == "B") or (s4 == "UB" and s2 == "UB"):
				self.error = "DiffAccuracyError: The constraint systems mus be different!."
			if len(s1) != self.length:
				self.error = "DiffAccuracyError: Constraint Length", len(s1), "is unequeal to Cstr length", self.length, "!"



	def checkFuzzyConstraintViolation(self, conformation_dotplot, index, fuzzy_request):
		"""
			Check if a requested Fuzzy Constraint definition area was already occupied by another Fuzzy Structure Definition
		"""
		i, j = index
		if (i, j) in self.SC[conformation_dotplot + "_FC"]:
			if self.SC[conformation_dotplot + "_FC"][(i, j)] != None:
				self.error = "Affected base pair ", (i, j), "has already been occupied by another Fuzzy Constraint Definition."
		else:
			self.SC[conformation_dotplot + "_FC"][(i, j)] = fuzzy_request


		# if (j, i) in self.SC[conformation_dotplot + "_FC"]:
		# 	if self.SC[conformation_dotplot + "_FC"][(j, i)] != None:
		# 		self.error = "Affected base pair ", (j, i), "has already been occupied by another Fuzzy Constraint Definition."
		# else:
		# 	self.SC[conformation_dotplot + "_FC"][(j, i)] = fuzzy_request


	def checkAccessibilityViolation(self, conformation_dotplot, index, accessibility_request):
		"""
			Check if a requested accessibility for a certain position is
			violating the already made constraints of that position.
		"""
		i = index
		if i in self.SC[conformation_dotplot + "_SS"]:
			if self.SC[conformation_dotplot + "_SS"][i][1] != None:
				self.error = "Position", i, "has already an affiliated accessibility"
			else: # accessibility == None
				accur_i, access_i = self.SC[conformation_dotplot + "_SS"][i]
				if 1-accur_i < accessibility_request:
					self.error = "Requested accessibility exceeds an already made accuracy setting..."
				self.SC[conformation_dotplot + "_SS"][i][1] = (accur_i, accessibility_request)
		else:
			self.SC[conformation_dotplot + "_SS"][i] = (None, accessibility_request)


	def checkAccuracyViolation(self, conformation_dotplot, index, accuracy_request):
		"""
			Check if a requested accuracy value for a certain position is
			violating the already made constraints of that position.
		"""
		i, j = index
		# CASES OF base pair i,j or j,i have already been occupied and therefore
		# will not allow further allocation in structure feature
		if (i, j) in self.SC[conformation_dotplot + "_BP"]:
			if self.SC[conformation_dotplot + "_BP"][(i, j)][0] != None:
				self.error = "Affected base pair ", (i, j), "has already been occupied by another accuracy constraint."
		if (j, i) in self.SC[conformation_dotplot + "_BP"]:
			if self.SC[conformation_dotplot + "_BP"][(j, i)][0] != None:
				self.error = "Affected base pair ", (j, i), "has already been occupied by another accuracy constraint."

		# CASE OF i
		if i not in self.SC[conformation_dotplot + "_SS"]: # Case of info for position i have not been collected yet.
			self.SC[conformation_dotplot + "_SS"][i] = (accuracy_request, None)
			self.SC[conformation_dotplot + "_BP"][(i, j)] = accuracy_request

		else: # Case of some info for position i have already been allocated
			accur_i, access_i = self.SC[conformation_dotplot+"_SS"][i]
			if accur_i != None and access_i != None:
				if accur_i + accuracy_request > 1:
					self.error = "StructureFeatureRequestError: Requested Accuracy exceeds limit of 1 and is not permitted to be added."
				if 1 - (accuracy_request + accur_i) < access_i:
					self.error = "StructureFeatureRequestError: Requested Accuracy undermines a constrained accessibility"
				self.SC[conformation_dotplot+"_SS"][i] = (accur_i + accuracy_request , access_i)
				self.SC[conformation_dotplot + "_BP"][(i, j)] = accuracy_request
			elif accur_i != None and access_i == None:
				if accur_i + accuracy_request > 1:
					self.error = "StructureFeatureRequestError: Requested Accuracy exceeds limit of 1 and is not permitted to be added."
				self.SC[conformation_dotplot+"_SS"][i] = (accur_i + accuracy_request , access_i)
				self.SC[conformation_dotplot + "_BP"][(i, j)] = accuracy_request

			elif accur_i == None and access_i != None:
				if 1 - accuracy_request < access_i:
					self.error = "StructureFeatureRequestError: Requested Accuracy undermines a constrained accessibility"
				self.SC[conformation_dotplot+"_SS"][i] = (accuracy_request , access_i)
				self.SC[conformation_dotplot + "_BP"][(i, j)] = accuracy_request

		# CASE OF j
		if j not in self.SC[conformation_dotplot + "_SS"]: # Case of info for position i have not been collected yet.
			self.SC[conformation_dotplot + "_SS"][j] = (accuracy_request, None)
			self.SC[conformation_dotplot + "_BP"][(j, i)] = accuracy_request

		else: # Case of some info for position i have already been allocated
			accur_j, access_j = self.SC[conformation_dotplot+"_SS"][j]
			if accur_j != None and access_j != None:
				if accur_j + accuracy_request > 1:
					self.error = "StructureFeatureRequestError: Requested Accuracy exceeds limit of 1 and is not permitted to be added."
				if 1 - (accuracy_request + accur_j) < access_j:
					self.error = "StructureFeatureRequestError: Requested Accuracy undermines a constrained accessibility"
				self.SC[conformation_dotplot+"_SS"][j] = (accur_j + accuracy_request , access_j)
				self.SC[conformation_dotplot + "_BP"][(j, i)] = accuracy_request

			elif accur_j != None:
				if accur_j + accuracy_request > 1:
					self.error = "StructureFeatureRequestError: Requested Accuracy exceeds limit of 1 and is not permitted to be added."
				self.SC[conformation_dotplot+"_SS"][j] = (accur_j + accuracy_request , access_j)
				self.SC[conformation_dotplot + "_BP"][(j, i)] = accuracy_request

			elif access_j != None:
				if (1 - accuracy_request) < access_j:
					self.error = "StructureFeatureRequestError: Requested Accuracy undermines a constrained accessibility"
				self.SC[conformation_dotplot+"_SS"][j] = (accuracy_request , access_j)
				self.SC[conformation_dotplot + "_BP"][(j, i)] = accuracy_request

	def check(self):
		"""
			CHECK THE COMMAND LINE STUFF
		"""

		self.print_to_STDOUT = (self.output_file == "STDOUT")

		if self.modus == "MFE":
			self.length = len(self.Cstr)
			if self.Cseq is None:
				self.Cseq = "N" * self.length
			self.parse_GC_management()


			self.checkForViennaTools()
			self.usedProgram = "RNAfold"
			if self.pseudoknots:
				if self.pkprogram == "pKiss":
					self.checkForpKiss()
					if self.error == "0" and self.pkparameter == True:
						self.alpha = 1.0
						self.beta = 0.1
						self.ER = 0.2
						self.Cstrweight = 0.1
						self.Cgcweight = 1.0
						self.Cseqweight = 0.5
						self.Cseqweight = 50
						self.ConvergenceCount = 100
						self.usedProgram = "pKiss"
				elif self.pkprogram == "HotKnots" and self.HotKnots_PATH != "":
					self.checkForHotKnots()
					if self.error == "0" and self.pkparameter == True:
						self.alpha = 1.0
						self.beta = 0.1
						self.ER = 0.2
						self.Cstrweight = 0.1
						self.Cgcweight = 1.0
						self.Cseqweight = 0.5
						self.Cseqweight = 50
						self.ConvergenceCount = 100
						self.usedProgram = "HotKnots"
				elif self.pkprogram == "IPKnot":
					self.checkForIPKnot()
					if self.error == "0" and self.pkparameter == True:
						self.alpha = 1.0
						self.beta = 0.1
						self.ER = 0.2
						self.Cstrweight = 0.1
						self.Cgcweight = 1.0
						self.Cseqweight = 0.5
						self.Cseqweight = 50
						self.ConvergenceCount = 100
						self.usedProgram = "IPKnot"
				else:
					self.error = " Please choose a suitable pseudoknot predictor: [pKiss|Hotknots|IPKnot]"

			# Constraint Checks and Parsing prior to Execution
			if self.error == "0":
				self.checkSimilarLength()
			if self.error == "0":
				self.isStructure()
			if self.error == "0":
				self.isBalanced()
			if self.error == "0":
				self.fulfillsHairpinRule()
			if self.error == "0":
				self.checkSequenceConstraint()
			if self.error == "0":
				self.parseExtendedVariables()
			if self.error == "0":
				self.checkConstaintCompatibility()



		elif self.modus == "DP":
			# # print "Access", self.accessibility
			# # print "Accur", self.accuracy
			# # print "Diff_access", self.diff_accessibility
			# # print "Diff_accur", self.diff_accuracy

			structurefeature_check = 0
			self.length = None
			if self.accessibility:
				if self.length == None:
					self.length = len(self.accessibility[0][0])
				self.check_Accessibilities()
				structurefeature_check += 1

			if self.accuracy:
				if self.length == None:
					self.length = len(self.accuracy[0][0])
				self.check_Accuracies()
				structurefeature_check += 1

			if self.diff_accessibility:
				if self.length == None:
					self.length = len(self.diff_accessibility[0][0])
				self.check_Diff_Accessibilities()
				structurefeature_check += 1

			if self.diff_accuracy:
				if self.length == None:
					self.length = len(self.diff_accuracy[0][0])
				self.check_Diff_Accuracies()
				structurefeature_check += 1

			if self.fuzzyCstr:
				if self.length == None:
					self.length = len(self.fuzzyCstr[0][0])
				self.check_FuzzyCstr()
				structurefeature_check += 1

			if self.diff_fuzzyCstr:
				if self.length == None:
					self.length = len(self.diff_fuzzyCstr[0][0])
				self.check_Diff_FuzzyCstr()
				structurefeature_check += 1


			if structurefeature_check == 0:
				self.error = "No structure constraint defined. Please define structural elements by using the accuracy of structure elements (basepairs) and accessibility of sequence stretches, fuzzy structure constraints are also a possibility."
			else:

				if self.Cseq == None:
					self.Cseq = self.length * "N"
				if self.error == "0":
					self.parse_GC_management()
				if self.error == "0":
					self.parseExtendedVariables()
				if self.error == "0":
					self.checkSequenceConstraint()
				if self.error == "0":
					self.parse_StructureFeatures()
				if self. error == "0":
					self.checkInterconnections()
				if self. error == "0":
					self.getPositionFeatures()



	def retrieveAllBasePairs(self):
		"""
			Extracts all base pairs from all made accuracy/diff_accuracy constrianits present.

		"""


		self.all_requested_BP = []
		if self.accur:
			for accur in self.accur:
				for i in accur[0]:
					if (i, accur[0][i]) not in self.all_requested_BP:
						self.all_requested_BP.append((i, accur[0][i]))
					if (accur[0][i], i) not in self.all_requested_BP:
						self.all_requested_BP.append((accur[0][i], i))
		if self.diffaccur:
			for diffaccur in self.diffaccur:
				for i in diffaccur[0]:
					if (i, diffaccur[0][i]) not in self.all_requested_BP:
						self.all_requested_BP.append((i, diffaccur[0][i]))
					if (diffaccur[0][i], i) not in self.all_requested_BP:
						self.all_requested_BP.append((diffaccur[0][i], i))

	def retrievePositionalInterconnection(self):
		"""
			Extract the interconnection information based on the accuracy constraints
			and the sequence constraint.
		"""
		self.interconnections = {}
		self.BPstack = {}
		self.LP = {}
		for i, j in self.all_requested_BP: # over all requested base pairs
			jj = (j, "".join(set(self.IUPAC[self.Cseq[j-1]]))) # translate sequence oncstraint into regular pnucleotide set

			if i not in self.interconnections: # inititalize an interaction
				self.interconnections[i] = [self.IUPAC[self.Cseq[i-1]], jj]
			else: # append an interaction to an already existing interaction constellation
				self.interconnections[i].append(jj)

		to_be_visited = self.interconnections.keys() # list of all already known source positions within base pair interactions constellations

		## print "Basis"
		#for i in tmp_intercon.keys():
			## print i, tmp_intercon[i]

		## print "\nSorted Basis"
		#index = sorted(self.interconnections,key=lambda k: len(self.interconnections[k][0]))
		#for i in index:
			## print i, self.interconnections[i]

		while len(to_be_visited) != 0: # visit all positions within base pair interaction constellations
			i = 0
			index = sorted(self.interconnections,key=lambda k: len(self.interconnections[k][0]))[i]
			while index not in to_be_visited:
				i += 1
				index = sorted(self.interconnections,key=lambda k: len(self.interconnections[k][0]))[i]

			## print "Index", index
			#ind = sorted(tmp_intercon,key=lambda k: len(tmp_intercon[k][0]))
			#for i in ind:
				## print i, tmp_intercon[i]
			## print "Index",index, tmp_intercon[index]
			## print [ self.IUPAC_reverseComplements[n] for n in self.interconnections[index][0]]
			new_nucs = "".join(set("".join([ self.IUPAC_reverseComplements[n] for n in self.interconnections[index][0]])))
			## print "->", new_nucs , "<-"


			for target_index in range(1, len(self.interconnections[index])):
				new_i, nucleotides = self.interconnections[index][target_index]
				if new_i in to_be_visited:
					if compareLists(new_nucs, self.interconnections[new_i][0]):
						self.interconnections[new_i][0] = new_nucs
						for ii in self.interconnections.keys():
							for id_new in range(1, len(self.interconnections[ii])):
								## print ii, id_new, self.interconnections[ii][id_new]
								if self.interconnections[ii][id_new][0] == new_i:
									## print self.interconnections[ii][id_new][1], new_nucs
									self.interconnections[ii][id_new] = (self.interconnections[ii][id_new][0], new_nucs)

			to_be_visited.pop(to_be_visited.index(index))

		self.Interconnection_sets = []
		for i in self.interconnections:
			indices_pre = []
			L_pre = 0
			indices_post = [int(i)]
			L_post = len(indices_post)
			o = 0
			while L_pre != L_post:
				for ii in indices_post:
					if int(ii) not in indices_pre:
						indices_pre.append(int(ii))
					for j in range(1, len(self.interconnections[ii])):
						if int(self.interconnections[ii][j][0]) not in indices_pre:
							indices_pre.append(int(self.interconnections[ii][j][0]))
				L_pre = len(indices_pre)
				L_post = len(indices_post)
				indices_post = indices_pre
				indices_pre = []
			s = set(indices_post)
			if s not in self.Interconnection_sets:
				self.Interconnection_sets.append(s)

	def detectSequenceConstraintViolation(self):

		for ig in self.interconnections:
			nucs = self.interconnections[ig][0]
			for i in range(1,len(self.interconnections[ig])):

				nucs = set(nucs)
				other_nucs = set(self.interconnections[ig][i][1])


				for n in nucs:
					iscompatible = False
					for nn in other_nucs:
						if isCompatible(n, nn, self.IUPAC_compatibles):
							iscompatible = True
					if iscompatible == False:
						self.error = "Found some nucleotide constraint which is opposing each other in Interconnection.%s %s" % (ig, self.interconnections[ig])

	def detect_circles(self):
		"""
			Detects cyrcles in secondary structure construct definitions, which have been preparsed into interconnections previously.
			Allows even sized cycles.
		"""
		self.interconnection_graphs = []

		## print self.Interconnection_sets
		for interconnection in self.Interconnection_sets:
			G = nx.Graph() # produce a new graph for an upcoming interaction set
			for arc in self.all_requested_BP: # loop over all requested base pairs
				if arc[0] in interconnection and arc[0] < arc[1]: # if an affiliated abse pair is in current interconnection

					G.add_edge(arc[0], arc[1]) # add to graph G
					G.node[arc[0]]["Cseq"] = self.interconnections[arc[0]][0]
					G.node[arc[1]]["Cseq"] = self.interconnections[arc[1]][0]
				if len(interconnection) > 2: # if it has at least two nodes
					cycles = nx.cycle_basis(G) # extract the number of cycles within the graph

					if cycles: # if it has
						for cycle in cycles:# iterate over cycles
							if len(cycle) % 2 != 0: # if they have an odd number, raise the error.
								self.error = "Found odd cycle on structure positions %s, of size %s" % (', '.join(str(e) for e in interconnection), cycle_length)

			self.interconnection_graphs.append(G)



		# # print self.Cseq
		# for g in  self.interconnection_graphs:
		# 	# print g.nodes()
		singletons = ""
		if len(self.interconnection_graphs) != 0:
			self.nodelist = set([])
			if len(self.interconnection_graphs) > 0:
				for graph in self.interconnection_graphs:
					self.nodelist = self.nodelist | set(graph.nodes())
			singletons = set([x for x in range(1,self.length + 1)]) - set(self.nodelist)
		else:
			singletons = [i for i in range(1, self.length + 1)]

		for s in list(singletons):
			G=nx.Graph()
			G.add_node(s)
			G.node[s]["Cseq"] = self.IUPAC[self.Cseq[s-1]]
			self.interconnection_graphs.append(G)
		# for g in  self.interconnection_graphs:
		# 	# print g.nodes(data=True)
		#exit(1)
		# # print self.nodelist
		# for g in self.interconnection_graphs:
		# 	# print g.nodes()
		# # 	for n in g.nodes(data=True):
		# # 		# print n
		# # 	# print g.edges()

		# exit(1)

	def checkInterconnections(self):
		"""
			Check all interconnections of base pairs on their integrity
		"""

		self.retrieveAllBasePairs()
		self.retrievePositionalInterconnection()
		self.detect_circles()
		self.detectSequenceConstraintViolation()

	def getPositionFeatures(self):
		"""
			Extracts for each position within the system, which constraint has
			been posed to that very position. According to this setup, for each
			position can be asked if it was important to the solution.
		"""

		### OPTION:
		### Make unconstrained positions as if they had been considered accessible
		self.PosFeatures = {}
		for i in range(1,self.length+1):
			self.PosFeatures[i] = {"UB":[], "B":[]}


		if self.accur:

			for elements in self.accur:
				## print elements
				for entry_key in elements[0].keys():
					entry_val = elements[0][entry_key]
					self.PosFeatures[entry_key][elements[1]].append(("Accu", entry_val,  elements[2]))

		if self.access:
			for elements in self.access:
				for entry_key in elements[0].keys():
					self.PosFeatures[entry_key][elements[1]].append(("Accs", entry_key, elements[2]))

		if self.diffaccur:
			for elements in self.diffaccur:
				for entry_key in elements[0].keys():
					entry_val = elements[0][entry_key]
					self.PosFeatures[entry_key][elements[1]].append(("Accu", entry_val, elements[2]))
					self.PosFeatures[entry_key][elements[3]].append(("Accu", entry_val, elements[4]))

		if self.diffaccess:
			for elements in self.diffaccess:
				for entry_key in elements[0].keys():
					self.PosFeatures[entry_key][elements[1]].append(("Accs", entry_key, elements[2]))
					self.PosFeatures[entry_key][elements[3]].append(("Accs", entry_key, elements[4]))



	def parse_StructureFeatures(self):
		"""
			Parse the structure features, such that each base, resp. base pair can
			only have a maximum shared probability of 1 in each dotplot.
			Only produce a [B]ound dotplot, if the -Cstr argument is present and
			if differential structure features or structure features within
			the [B]ound dotplot are mentioned
		"""

		# Lists of the nucleotide positions, each position is allowed to have a
		# maximum probability of one to be paired and a probability of being
		# unpaired (1-x).
		self.SC = {"B_BP":{}, "UB_BP":{}, "B_SS":{}, "UB_SS":{}, "B_FC":{}, "UB_FC":{}}

		# Check for each requested base pair or accessibility if each newly added
		# requested feature is compatible with the already inputed ones.
		# Initially all positions are accessible. A structure requests lowers the resp. accessibility.

		# ACCESSIBILITY
		self.access = []
		if self.accessibility:
			for request in self.accessibility:
				struct_info = removePairedAndUndefined_From_bpstack(request[0] , getbpStack(len(request[0]) * ".") [0])
				constraint_system = request[1]
				value = request[2]
				self.addAccessibilityRequest(struct_info, constraint_system, value)
				self.access.append((struct_info, constraint_system, value))

		# DIFF_ACCESSIBILITY
		self.diffaccess = []
		if self.diff_accessibility:
			for request in self.diff_accessibility:
				struct_info = removePairedAndUndefined_From_bpstack(request[0] , getbpStack(len(request[0]) * ".") [0])
				constraint_system_1 = request[1]
				value_1 = request[2]
				constraint_system_2 = request[3]
				value_2 = request[4]
				self. addDiffAccessibilityRequest(struct_info , constraint_system_1, value_1, constraint_system_2, value_2)
				self.diffaccess.append((struct_info , constraint_system_1, value_1, constraint_system_2, value_2))


		self.accur = []
		# CONSTRAINT STRUCTURE
		if self.Cstr != None:
			self.addAccuracyRequest(removeUnpairedFrom_bpstack(getbpStack(self.Cstr)[0]), "B", 1)
			self.accur.append((removeUnpairedFrom_bpstack(getbpStack(self.Cstr)[0]), "B", 1))

		# ACCURACY

		if self.accuracy:
			for request in self.accuracy:
				struct_info = removeUnpairedFrom_bpstack(getbpStack(request[0])[0])
				constraint_system_1 = request[1]
				value_1 = request[2]
				self.addAccuracyRequest(struct_info, constraint_system_1, value_1)
				self.accur.append((struct_info, constraint_system_1, value_1))

		# DIFF_ACCURACY
		self.diffaccur = []
		if self.diff_accuracy:
			for request in self.diff_accuracy:
				struct_info = removeUnpairedFrom_bpstack(getbpStack(request[0])[0])
				constraint_system_1 = request[1]
				value_1 = request[2]
				constraint_system_2 = request[3]
				value_2 = request[4]
				self.addDiffAccuracyRequest(struct_info, constraint_system_1, value_1, constraint_system_2, value_2)
				self.diffaccur.append((struct_info, constraint_system_1, value_1, constraint_system_2, value_2))

		# Fuzzy Structure Constrint fCstr
		self.fuzzy = []
		if self.fuzzyCstr:
			for request in self.fuzzyCstr:
				struct_info = retrieveFuzzyStack(request[0])
				constraint_system_1 = request[1]
				value_1 = request[2]
				self.addFuzzyConstraintFeature(struct_info, constraint_system_1, value_1)
				self.fuzzy.append((struct_info, constraint_system_1, value_1))

		# Differential Fuzzy Structure Constriant dfCstr
		self.difffuzzy = []
		if self.diff_fuzzyCstr:
			for request in self.diff_fuzzyCstr:
				struct_info = retrieveFuzzyStack(request[0])
				constraint_system_1 = request[1]
				value_1 = request[2]
				constraint_system_2 = request[3]
				value_2 = request[4]

				self.addDiffFuzzyConstraintFeature(struct_info, constraint_system_1, value_1, constraint_system_2, value_2)
				self.difffuzzy.append((struct_info, constraint_system_1, value_1, constraint_system_2, value_2))



	def addFuzzyConstraintFeature(self, struct_info, constraint_system_1, value_1):
		"""
			Adds a fuzzy structure feature to the the respective dotplot
		"""
		for i in struct_info:
			self.checkFuzzyConstraintViolation(constraint_system_1, i, value_1)

	def addDiffFuzzyConstraintFeature(self, struct_info, constraint_system_1, value_1, constraint_system_2, value_2):
		"""
			Adds a fuzzy structure feature to the the respective dotplot
		"""
		for i in struct_info:
			self.checkFuzzyConstraintViolation(constraint_system_1, i, value_1)
			self.checkFuzzyConstraintViolation(constraint_system_2, i, value_2)


	def addAccuracyRequest(self, struct_info, constraint_system_1, value_1):
		"""
			Adds accuracy structure elements to the respective conformation dotplot.
		"""
		## print type(struct_info), type(constraint_system_1), type(value_1)
		for i in struct_info:
			j = struct_info[i]
			self.checkAccuracyViolation(constraint_system_1, (i, j), value_1)

	def addDiffAccuracyRequest(self, request_index, conformation_dotplot_1, accuracy_request_1, conformation_dotplot_2, accuracy_request_2):
		"""
			Adds differential accuracy elements to both conformation dotplots
		"""
		for i in request_index:
			j = request_index[i]
			self.checkAccuracyViolation(conformation_dotplot_1, (i, j), accuracy_request_1)
			self.checkAccuracyViolation(conformation_dotplot_2, (i, j), accuracy_request_2)


	def addAccessibilityRequest(self, request_index, conformation_dotplot, accessibility_request):
		"""
			Adds accessibility  structure element to the respective conformation dotplot
		"""
		for i in request_index:
			self.checkAccessibilityViolation(conformation_dotplot, i, accessibility_request)

	def addDiffAccessibilityRequest(self, request_index, conformation_dotplot_1, accessibility_request_1, conformation_dotplot_2, accessibility_request_2):
		"""
			Adds a differential accessibility to both conformation dotplots
		"""
		for i in request_index:
			self.checkAccessibilityViolation(conformation_dotplot_1, i, accessibility_request_1)
			self.checkAccessibilityViolation(conformation_dotplot_2, i, accessibility_request_2)

	def parseExtendedVariables(self):
		"""
			Parse aux variables and load IUPAC
		"""
		if self.seed != "none":
			random.seed(self.seed)

		if self.py == False:
			if self.print_to_STDOUT == False:
				outfolder = '/'.join(self.output_file.strip().split("/")[:-1])
				curr_dir = os.getcwd()
				if not os.path.exists(outfolder):
					os.makedirs(outfolder)
				os.chdir(outfolder)

		self.transform()

		# Allowed deviation from the structural target:
		self.objective_to_target_distance = 0.0

		# Loading the IUPAC copatibilities of nuleotides and their abstract representing symbols
		self.IUPAC = {"A":"A", "C":"C", "G":"G", "U":"U", "R":"AG", "Y":"CU", "S":"GC", "W":"AU","K":"GU", "M":"AC", "B":"CGU", "D":"AGU", "H":"ACU", "V":"ACG", "N":"ACGU"}
		self.IUPAC_compatibles = loadIUPACcompatibilities(self)

		# IUPAC Management
		if self.noGUBasePair == True: ## Without the GU basepair
			self.IUPAC_reverseComplements = {"A":"U", "C":"G", "G":"C", "U":"A", "R":"UC", "Y":"AG", "S":"GC", "W":"UA","K":"CA", "M":"UG", "B":"AGC", "D":"ACU", "H":"UGA", "V":"UGC", "N":"ACGU"}
		else: ## allowing the GU basepair
			self.IUPAC_reverseComplements = {"A":"U", "C":"G", "G":"UC", "U":"AG", "R":"UC", "Y":"AG", "S":"UGC", "W":"UAG","K":"UCAG", "M":"UG", "B":"AGCU", "D":"AGCU", "H":"UGA", "V":"UGC", "N":"ACGU"}

		if self.modus == "DP":
			if self.trailblazer > 1.0 and self.trailblazer < 0:
				self.error = "The trail blaze threshold should be within [0,1]."

	def reachableGC(self):
		"""
			Checks if a demanded GC target content is reachable in dependence with the given sequence constraint.
			For each explicit and ambiguous character definition within the constraint, the respective possibilities
			are elicited: "A" counts as "A", but for "B", also an "U" is possible, at the same time, "G" or "C" are possible
			as well. So two scenarios can be evaluated: A minimum GC content, which is possible and a maximum GC content.
			For this, for all not "N" letters their potential towards  their potentials is evaluated and counted in the
			respective counters for min and max GC content. Only those characters are taken into concideration which would enforce
			a categorial pyrimidine/purine decision. (ACGU, SW)

		"""

		nucleotide_contribution = 1/float(self.length)

		minGC = 0.0
		maxGC = 1.0
		for i in self.Cseq:
			if i != "N":
				if i == "A" or i == "U":
					maxGC -= nucleotide_contribution
				elif i == "C" or i == "G":
					minGC += nucleotide_contribution
				elif i == "S":#(G or C)
					minGC += nucleotide_contribution
				elif i == "W":#(A or T/U):
					maxGC -= nucleotide_contribution
		return ('%.6f' % (minGC), '%.6f' % (maxGC))


	def parse_GC_management(self):
		"""
			Parse and and transform tGC values
		"""

		if len(self.tGC) == 1 and type(self.tGC[0]) is float: # CASE Only one tGC value is defined, which needs to account for the whole terrain
			tgc = self.tGC.pop()
			self.tGC.append((tgc, 0, self.length - 1))

		for t in self.tGC:
			if len(t) != 3:
				self.error = "Error :: Not enough tGC and affiliated areas declarations"

		if self.error == "0":
			check_set = set(range(0, self.length ))

			curr_set = set()
			for i, area in enumerate(self.tGC): # CHECK if the areas are consistent and do not show disconnectivity.
				## print i, area
				v, s1, s2 = area
				if v < 0 or v > 1:
					self.error = "Error: Chosen tGC > %s < not in range [0,1]" % (i)

				if self.error == "0":
					tmp_set = set(range(int(s1), int(s2 + 1)))
					## print "curr_set", curr_set
					## print "tmp_set", tmp_set
					if len(curr_set.intersection(tmp_set)) == 0:
						curr_set = curr_set.union(tmp_set)
					else:
						self.error = "Error: Double defined tGC declaration area sector detected. Nucleotide positions", ", ".join(str(e) for e in curr_set.intersection(tmp_set)), "show(s) redundant tGC declaration"

		if self.error == "0":
			## print "Curr", curr_set
			## print "Check", check_set
			if len(curr_set.symmetric_difference(check_set)) != 0:
				self.error = "Error: Undefined tGC area sectors detected. Nucleotide positions", ", ".join(str(e) for e in curr_set.symmetric_difference(check_set)), "is/are not covered."

		for tgc in self.tGC: # CHECK if the specified GC values can be reached at all...
			if self.error == "0":
				v, start, stop = tgc
				v = '%.6f' % (v)
				# tmp_sc = self.Cseq[start:stop + 1]
				minGC, maxGC = self.reachableGC()
				if v > maxGC or v < minGC:
					self.error = "WARNING: Chosen target GC %s content is not reachable. The selected sequence constraint contradicts the tGC constraint value. Sequence Constraint allows tGC only to be in [%s,%s]" % (v, minGC, maxGC)


	##########################
	# PROGRAM PRESENCE CHECK
	##########################

	def checkForViennaTools(self):
		"""
		Checking for the presence of the Vienna tools in the system by which'ing for RNAfold and RNAdistance
		"""
		RNAfold_output = subprocess.Popen(["which", "RNAfold"], stdout=subprocess.PIPE).communicate()[0].strip()
		if not (len(RNAfold_output) > 0 and RNAfold_output.find("found") == -1 and RNAfold_output.find(" no ") == -1):
			self.error = "No RNAfold found\nIt seems the Vienna RNA Package is not installed on your machine. Please do so!\nYou can get it at http://www.tbi.univie.ac.at/"

	def checkForpKiss(self):
		"""
			Checking for the presence of pKiss
		"""
		pKiss_output = subprocess.Popen(["which", "pKiss_mfe"], stdout=subprocess.PIPE).communicate()[0].strip()
		#pKiss_output = subprocess.Popen(["which", "/usr/local/pkiss/2014-03-17/bin/pKiss_mfe"], stdout=subprocess.PIPE).communicate()[0].strip()
		if not (len(pKiss_output) > 0 and pKiss_output.find("found") == -1 and pKiss_output.find(" no ") == -1):
			self.error = "No pKiss found\nIt seems that pKiss is not installed on your machine. Please do so!\nYou can get it at http://bibiserv2.cebitec.uni-bielefeld.de/pkiss"

	def checkForIPKnot(self):
		"""
			Checking for the presence of IPKnot
		"""
		IPKnot_output = subprocess.Popen(["which", "ipknot"], stdout=subprocess.PIPE).communicate()[0].strip()
		if not (len(IPKnot_output) > 0 and IPKnot_output.find("found") == -1 and IPKnot_output.find(" no ") == -1):
			self.error = "No IPKnot\nIt seems that IPKnot is not installed on your machine. Please do so!\nYou can get it at http://rtips.dna.bio.keio.ac.jp/ipknot/"

	def checkForHotKnots(self):
		"""
			Checking for the presence of HotKnots
		"""
		cmd = self.HotKnots_PATH + "/HotKnots"
		if not os.path.exists(cmd):
			self.error = "HotKnots\nIt seems that HotKnots is not installed on your machine. Please do so!\nYou can get it at http://www.cs.ubc.ca/labs/beta/Software/HotKnots/"


#########################
# ENTRY TO THE ANT HIVE
#########################

if __name__ == "__main__":

	exe()
