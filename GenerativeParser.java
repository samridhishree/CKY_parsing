package edu.berkeley.nlp.assignments.parsing.student;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import edu.berkeley.nlp.assignments.parsing.BinaryRule;
import edu.berkeley.nlp.assignments.parsing.Grammar;
import edu.berkeley.nlp.assignments.parsing.Parser;
import edu.berkeley.nlp.assignments.parsing.SimpleLexicon;
import edu.berkeley.nlp.assignments.parsing.UnaryClosure;
import edu.berkeley.nlp.assignments.parsing.UnaryRule;
import edu.berkeley.nlp.ling.Tree;
import edu.berkeley.nlp.util.Indexer;

/**
 * @author Samridhi
 * This class implements a Generative Parser trained on the 
 * provided training trees. It uses the CKY algorithm which
 * produces parse trees with alternating Unary and Binary rules.
 * The training trees used have been markovized with h=2, v=2
 * horizontal and vertical markovization.
 *
 */
public class GenerativeParser implements Parser 
{
	SimpleLexicon lexicon;
	double[][][] scoreUnary;
	double[][][] scoreBinary;
	Indexer<String> labelIndexer;
	BinaryBackPointer[][][] bpBinary;
	List<Integer> binaryLeftChildren;
	int[][][] bpUnary;
	Grammar grammar;
	UnaryClosure uc;
	int nonTerminalSize;
	
	public GenerativeParser(List<Tree<String>> trainTrees) 
	{
		System.out.print("Annotating / binarizing training trees ... ");
	    List<Tree<String>> annotatedTrainTrees = AnnotateTrees(trainTrees);
	    System.out.println("done.");
	    System.out.print("Building grammar ... ");
	    grammar = Grammar.generativeGrammarFromTrees(annotatedTrainTrees);
	    System.out.println("done. (" + grammar.getLabelIndexer().size() + " states)");
	    
	    
	    // Train word-tag scores
	    lexicon = new SimpleLexicon(annotatedTrainTrees);
	    labelIndexer = grammar.getLabelIndexer();
	    
	    //Get the binary left children
	    Set<Integer> tempSet = new HashSet<Integer>();
	    for(BinaryRule rule : grammar.getBinaryRules())
	    {
	    	tempSet.add(rule.getLeftChild());
	    }
	    binaryLeftChildren = new ArrayList<Integer>(tempSet);
	    
	    //Compute Unary Closure
	    uc = new UnaryClosure(labelIndexer, grammar.getUnaryRules());
	    nonTerminalSize = labelIndexer.size();
	}

	 // Returns the most probable parse tree for a given sentence
	 public Tree<String> getBestParse(List<String> sentence)
	 {
		 //Compute and fill the unary/binary charts and update backpointers
		 ComputeBestScoreChart(sentence);
		 int n = sentence.size();
		 int rootIndex = labelIndexer.addAndGetIndex("ROOT");
		 Tree<String> bestParseTree;
		 
		 if(Double.isInfinite(scoreUnary[0][n][rootIndex]))
			 bestParseTree = new Tree<String>("ROOT", Collections.singletonList(new Tree<String>("JUNK")));
		 else
			 bestParseTree = CreateBestParseTree(sentence, "ROOT", 0, n, true);
		 return TreeAnnotationBinarization.unAnnotateTree(bestParseTree);
	}
	 
	 private List<Tree<String>> AnnotateTrees(List<Tree<String>> trees) 
	 {
	    List<Tree<String>> annotatedTrees = new ArrayList<Tree<String>>();
	    for (Tree<String> tree : trees) 
	    {
	      annotatedTrees.add(TreeAnnotationBinarization.AnnotateTreeMarkovizedBinarization(tree));
	    }
	    return annotatedTrees;
	 }
	 
	 private void ComputeBestScoreChart(List<String> sentence)
	 {
		 int n = sentence.size();
		 int i,j,k,diff;
		 int parent, rightChild, child;
		 
		 //Initialize the scoreUnary,scoreBinary and backpointer arrays
		 scoreUnary = new double[n+1][n+1][nonTerminalSize];
		 scoreBinary = new double[n+1][n+1][nonTerminalSize];
		 bpBinary = new BinaryBackPointer[n+1][n+1][nonTerminalSize];
		 bpUnary = new int[n+1][n+1][nonTerminalSize];
		 
		 //Initialize score arrays
		 for(i=0; i<=n; i++)
		 {
			 for(j=0; j<=n; j++)
			 {
				 for(k=0; k<nonTerminalSize; k++)
				 {
					 scoreUnary[i][j][k] = Double.NEGATIVE_INFINITY;
					 scoreBinary[i][j][k] = Double.NEGATIVE_INFINITY;
					 bpUnary[i][j][k] = Integer.MIN_VALUE;
				 }
			 }
		 }
		 
		 //Update (word|tag) scores
		 for(i=0; i<n ; i++)
		 {
			 for(int nonTerminal = 0; nonTerminal < nonTerminalSize; nonTerminal++)
			 {
				 String word = sentence.get(i);
				 String label = labelIndexer.get(nonTerminal);
				 Double tempScore = lexicon.scoreTagging(word, label);
				 if(!(Double.isNaN(tempScore)) &&
				    !(Double.isInfinite(tempScore)))
				 {
					 scoreUnary[i][i+1][nonTerminal] = tempScore;
				 }
			 }
		 }
		 UpdatePreTerminalUnaryScores(sentence);
		 
		 //Alternate chart updation for unary and binary rules
		 for(diff=2; diff<=n; diff++)
		 {
			 for(i=0; i<=(n-diff); i++)
			 {
				 j = i+diff;
				 for(k=(i+1); k<=(j-1);  k++)
				 {
					 for(int nonTerm : binaryLeftChildren)
					 {
						 if((scoreUnary[i][k][nonTerm] != Double.NEGATIVE_INFINITY))
						 {
							for(BinaryRule rule : grammar.getBinaryRulesByLeftChild(nonTerm))
							{
								parent = rule.getParent();
								rightChild = rule.getRightChild();
							 
								double tempScore = (rule.getScore() + 
									   				scoreUnary[i][k][nonTerm] + 
									   				scoreUnary[k][j][rightChild]);
							 	if(scoreBinary[i][j][parent] < tempScore)
								 {
									 scoreBinary[i][j][parent] = tempScore;
									 bpBinary[i][j][parent] = new BinaryBackPointer();
									 bpBinary[i][j][parent].parent = labelIndexer.get(parent);
									 bpBinary[i][j][parent].lChild = labelIndexer.get(nonTerm);
									 bpBinary[i][j][parent].rChild = labelIndexer.get(rightChild);
									 bpBinary[i][j][parent].splitPoint = k;
								 }
							   }
					 		}
					 	}
					 }
				 
				 for(int nonTerm = 0; nonTerm < nonTerminalSize; nonTerm++)
				 {
					 if(!Double.isInfinite(scoreBinary[i][j][nonTerm]))
					 {
						 Double tempScore;
						 for(UnaryRule rule : uc.getClosedUnaryRulesByChild(nonTerm))
						 {
							 parent = rule.getParent();
							 child = rule.getChild();
							 tempScore = rule.getScore() + scoreBinary[i][j][child];
						 
							 if(scoreUnary[i][j][parent] < tempScore)
							 {
								 scoreUnary[i][j][parent] =  tempScore;
								 bpUnary[i][j][parent] = child;
							 }
						 }
					 }
				 }
			 }
		 }
	 }
	 
	 /*
	  * Updates the score for the parent of any pre-terminal unary rule that results in the word (span = 1)
	  */
	 private void UpdatePreTerminalUnaryScores(List<String> sentence)
	 {
		 int parent, child;
		 int n = sentence.size();
		 Double tempScore;
		
		 for(int i=0; i<n; i++)
		 {
			 for(int nonTerm = 0; nonTerm < nonTerminalSize; nonTerm++)
			 {
				 if(!Double.isInfinite(scoreUnary[i][i+1][nonTerm]))
				 {
					 for(UnaryRule rule : uc.getClosedUnaryRulesByChild(nonTerm))
					 {
						 parent = rule.getParent();
						 child = rule.getChild();
						 tempScore = scoreUnary[i][i+1][child] + rule.getScore();
						 
						 //Update score for [i][i+1][parent] for a possible unary derivation parent->child->word
						 if(scoreUnary[i][i+1][parent] < tempScore)
						 {
							 scoreUnary[i][i+1][parent] = tempScore;
							 bpUnary[i][i+1][parent] = child;
						 }
					 }
				 }
			 }
		 }
	 }
	 
	 /*
	  * Creates the parse tree from the back pointers stored
	  */
	 private Tree<String> CreateBestParseTree(List<String> sentence, String parentLabel, 
			 							int start, int end, Boolean useUnary)
	 {
		 int parentIndex = labelIndexer.addAndGetIndex(parentLabel);
		 
		 //Base Case
		 if(end == start+1)
		 {
			 String word = sentence.get(start);
			 Tree<String> parentNode = null;
			 String childLabel;
			 
			 //If backpointer is present, get the preterminal and append word
			 if((bpUnary[start][end][parentIndex] != Integer.MIN_VALUE))
			 {
				 childLabel = labelIndexer.get(bpUnary[start][end][parentIndex]);
				 UnaryRule tempRule = new UnaryRule(parentIndex, labelIndexer.addAndGetIndex(childLabel));
				 List<Integer> closurePath = uc.getPath(tempRule);
				
				 if(closurePath != null && closurePath.size() > 2)
				 {
					 int n = closurePath.size();
					 //Create the last node with the terminal word and then expand the rule above
					 Tree<String> terminalNode = new Tree<String>(word);
					 List<Tree<String>> tempList = new ArrayList<Tree<String>>();
					 tempList.add(terminalNode);
					 Tree<String> lastChildNode = new Tree<String>(labelIndexer.get(closurePath.get(n-1)), tempList);
					 
					 //Expand the unary closure rules above this node
					 parentNode = ExpandUnaryClosure(closurePath, lastChildNode);
				 }
				 else
				 {
					 Tree<String> leafNode = new Tree<String>(word);
					 List<Tree<String>> tempListChildren = new ArrayList<Tree<String>>();
					 tempListChildren.add(leafNode);
					 Tree<String> preTerminalNode = new Tree<String>(childLabel, tempListChildren);
					 List<Tree<String>> preTerminalList = new ArrayList<Tree<String>>();
					 preTerminalList.add(preTerminalNode);
					 parentNode = new Tree<String>(parentLabel, preTerminalList);
				 } 
			 }
			 else
			 {
				 //Either it is a pre-terminal or it is an identity rule
				 Tree<String> leafNode = new Tree<String>(word);
				 parentNode = new Tree<String>(parentLabel, Collections.singletonList(leafNode));
			}
			 return parentNode;
		 }
		 
		 if(useUnary)  
		 {
			String child = labelIndexer.get(bpUnary[start][end][parentIndex]);
			Tree<String> parentTree;
			
			//Handle reflexive rule
			if(parentLabel.equals(child))
			{
				//Directly create binary split on this parent
				parentTree = CreateBestParseTree(sentence, parentLabel, start, end, false);
			}
			else
			{
				UnaryRule tempRule = new UnaryRule(parentIndex, labelIndexer.addAndGetIndex(child));
				List<Integer> closurePath = uc.getPath(tempRule);
				
				if(closurePath != null && (closurePath.size() > 2))
				{
					int n = closurePath.size();
					Tree<String> lastUnaryNode = CreateBestParseTree(sentence, labelIndexer.get(closurePath.get(n-1)), start, end, false);
					parentTree = ExpandUnaryClosure(closurePath, lastUnaryNode);
				}
				else
				{
					Tree<String> childNode = CreateBestParseTree(sentence, child, start, end, false);
					List<Tree<String>> children = new ArrayList<Tree<String>>();
					children.add(childNode);
					parentTree = new Tree<String>(parentLabel, children);
				}
			}
			return parentTree; 
		 }
		 else
		 {
			 int k = bpBinary[start][end][parentIndex].splitPoint;
			 String lChild = bpBinary[start][end][parentIndex].lChild;
			 String rChild = bpBinary[start][end][parentIndex].rChild;
			 
			 List<Tree<String>> children = new ArrayList<Tree<String>>();
			 Tree<String> leftChildNode = CreateBestParseTree(sentence, lChild, start, k, true);
			 Tree<String> rightChildNode = CreateBestParseTree(sentence, rChild, k, end, true);
			 children.add(leftChildNode);
			 children.add(rightChildNode);
			 Tree<String> parentTree = new Tree<String>(bpBinary[start][end][parentIndex].parent, children);
			
			 return parentTree;
		 }
	 }
	 
	 private Tree<String> ExpandUnaryClosure(List<Integer> nonTerminalPath, Tree<String> childTree)
	 {
		 Tree<String> rootNode = childTree;
		 Tree<String> tempChild = childTree;
		 
		 
		 for(int i = (nonTerminalPath.size() - 2); i>=0; i--)
		 {
			 List<Tree<String>> childrenList = new ArrayList<Tree<String>>();
			 childrenList.add(tempChild);
			 Tree<String> tempParent = new Tree<String>(labelIndexer.get(nonTerminalPath.get(i)), childrenList);
			 tempChild = tempParent;
			 rootNode = tempParent;
		 }
		 return rootNode;
 	 }
	 
	 /*
	  * Back Pointer classes for unary and binary
	  */
	 public class BinaryBackPointer
	 {
	 	public String parent = new String();
	 	public String lChild = new String();
	 	public String rChild = new String();
	 	public int splitPoint = -1;
	 }
}