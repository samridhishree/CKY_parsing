package edu.berkeley.nlp.assignments.parsing.student;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import edu.berkeley.nlp.ling.Tree;
import edu.berkeley.nlp.ling.Trees;
import edu.berkeley.nlp.util.Filter;

/**
 * Class which contains code for annotating and binarizing trees for the
 * parser's use, and debinarizing and unannotating them for scoring.
 */
public class TreeAnnotationBinarization
{
	/**
	 * This performs lossy binarization. It performs horizontal and
	 * vertical markovization with parameters: v=2 and h=2
	 * 
	 * @param unAnnotatedTree
	 * @return
	 */
	public static Tree<String> AnnotateTreeMarkovizedBinarization(Tree<String> unAnnotatedTree) 
	{
		return BinarizeTree(unAnnotatedTree);
	}

	private static Tree<String> BinarizeTree(Tree<String> tree) 
	{
		String label = tree.getLabel();
		if (tree.isLeaf()) 
			return new Tree<String>(label);
		
		// For vertical markovization, annotate the children labels with parent label if it is not a leaf
		tree = VerticalAnnotation(tree);
		
		
		if (tree.getChildren().size() == 1) 
		{ 
			return new Tree<String>(label, Collections.singletonList(BinarizeTree(tree.getChildren().get(0)))); 
		}
		
		// otherwise, it's a binary-or-more local tree, so decompose it into a sequence of binary and unary trees.
		String intermediateLabel = "@" + label + "->_";
		Tree<String> intermediateTree = BinarizeTreeHelper(tree, 0, intermediateLabel);
		return new Tree<String>(label, intermediateTree.getChildren());
	}

	private static Tree<String> BinarizeTreeHelper(Tree<String> tree, int numChildrenGenerated, String intermediateLabel)
	{
		Tree<String> leftTree = tree.getChildren().get(numChildrenGenerated);
		List<Tree<String>> children = new ArrayList<Tree<String>>();
		children.add(BinarizeTree(leftTree));
		
		if (numChildrenGenerated < tree.getChildren().size() - 1) 
		{
			String newLabel = HorizontalMarkovization(intermediateLabel, leftTree.getLabel());
			Tree<String> rightTree = BinarizeTreeHelper(tree, numChildrenGenerated + 1, newLabel);
			children.add(rightTree);
		}
		return new Tree<String>(intermediateLabel, children);
	}
	

	public static Tree<String> unAnnotateTree(Tree<String> annotatedTree) 
	{
		// Remove intermediate nodes (labels beginning with "@"
		// Remove all material on node labels which follow their base symbol (cuts anything after <,>,^,=,_ or ->)
		// Examples: a node with label @NP->DT_JJ will be spliced out, and a node with label NP^S will be reduced to NP
		Tree<String> debinarizedTree = Trees.spliceNodes(annotatedTree, new Filter<String>()
		{
			public boolean accept(String s) {
				return s.startsWith("@");
			}
		});
		Tree<String> unAnnotatedTree = (new Trees.LabelNormalizer()).transformTree(debinarizedTree);
		return unAnnotatedTree;
	}
	
	/*
	 * Performs v=2 vertical markovization of the tree (marks each child label with its immediate parent)
	 */
	private static Tree<String> VerticalAnnotation(Tree<String> tree)
	{
		Tree<String> verticalMarkovizedTree = tree;
		String label = verticalMarkovizedTree.getLabel();
		String[] labelPart = label.split("\\^");
		String unAnnotatedParentLabel = labelPart[0];
		unAnnotatedParentLabel = unAnnotatedParentLabel.replace("@", "").trim();
		String childLabel;
		
		for(Tree<String> child : verticalMarkovizedTree.getChildren())
		{
			if(child.isPreTerminal() || child.isPhrasal())
			{
				childLabel = child.getLabel().concat("^" + unAnnotatedParentLabel);
				child.setLabel(childLabel);
			}
		}
		return verticalMarkovizedTree;
	}
	
	/*
	 * Performs h=2 horizontal markovization
	 */
	private static String HorizontalMarkovization(String label, String siblingLabel)
	{
		String finalLabel = "";
		siblingLabel = siblingLabel.split("\\^")[0].trim();
		
		if(label.indexOf('_') != -1)
		{
			String[] labelPart = label.split("_");
			if(labelPart.length > 2)
			{
				//Rule of the form X^Y->_Z_W. Remove Z and concatenate sibling
				finalLabel = labelPart[0] + "_" + labelPart[2] + "_" + siblingLabel; 
			}
			else
			{
				if(labelPart.length == 1)
					//Rule of the form X^Y->_. Just concatenate the sibling label
					finalLabel = label.concat(siblingLabel);
				else
					//Rule of the form X^Y->_Z. Concatenate _SiblingLabel
					finalLabel = label.concat("_" + siblingLabel);
			}
		}
		return finalLabel;
	}
}