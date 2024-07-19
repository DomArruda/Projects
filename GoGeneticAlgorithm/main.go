/*
Genetic Algorithm for Symbolic Regression

This algorithm uses a genetic approach to evolve mathematical expressions that best fit a given dataset.
Here's a detailed explanation of how it works:

1. Representation:
   - Expressions are represented as binary trees (Node struct).
   - Leaf nodes are either variables or constants.
   - Non-leaf nodes are mathematical operations (addition, subtraction, multiplication, division, sine, cosine).

2. Initialization:
   - A population of random expression trees is generated.
   - Trees are created with a maximum depth to control initial complexity.

3. Fitness Evaluation:
   - For each individual in the population:
     a. The expression tree is evaluated against the input data.
     b. Mean Squared Error (MSE) between predicted and actual values is calculated.
     c. Tree complexity (number of nodes) is calculated.
     d. Adjusted fitness combines MSE and a complexity penalty to favor simpler expressions.

4. Selection:
   - Tournament selection is used to choose parents for the next generation.
   - A small subset of individuals is randomly selected, and the best among them becomes a parent.

5. Crossover:
   - Two parent trees exchange random subtrees to create two child trees.

6. Mutation:
   - Random changes are applied to the tree structure:
     - Changing operation types
     - Replacing subtrees with new random subtrees
     - Modifying constant values
     - Changing variable nodes

7. Elitism:
   - The best individuals from the current generation are preserved unchanged in the next generation.

8. Termination:
   - The algorithm runs for a fixed number of generations or until a satisfactory solution is found.

9. Concurrency:
   - Population evaluation and evolution are performed concurrently to improve performance.

The algorithm balances between finding accurate expressions (low MSE) and maintaining simplicity
(low complexity) through the adjusted fitness score. This approach helps in discovering
meaningful and generalizable relationships in the data rather than overfitting.
*/

package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Define the operation types
const (
	ADD = iota
	SUB
	MUL
	DIV
	SIN
	COS
	POW
	LOG
)

// Node represents a node in the expression tree
type Node struct {
	Op       int
	Value    float64
	Left     *Node
	Right    *Node
	IsLeaf   bool
	Variable int // Index of the variable in the input array
}

// Individual represents a solution in the population
type Individual struct {
	Tree            *Node
	Fitness         float64
	Complexity      int
	AdjustedFitness float64
}

// Population is a slice of individuals
type Population []*Individual

func Evaluate(node *Node, x []float64) float64 {
	if node == nil {
		return 0
	}
	if node.IsLeaf {
		if node.Variable >= 0 && node.Variable < len(x) {
			return x[node.Variable]
		}
		return node.Value
	}

	left := Evaluate(node.Left, x)

	// Only evaluate right child for binary operations
	var right float64
	if node.Op < LOG {
		right = Evaluate(node.Right, x)
	}

	switch node.Op {
	case ADD:
		return left + right
	case SUB:
		return left - right
	case MUL:
		return left * right
	case DIV:
		if math.Abs(right) < 1e-10 {
			return 1 // Avoid division by zero
		}
		return left / right
	case SIN:
		return math.Sin(left)
	case COS:
		return math.Cos(left)
	case POW:
		// Avoid potential overflow or domain errors
		if left < 0 && math.Floor(right) != right {
			return 0 // Return 0 for negative base with non-integer exponent
		}
		if math.Abs(right) > 100 {
			return 0 // Avoid potential overflow
		}
		return math.Pow(left, right)
	case LOG:
		if left <= 0 {
			return 0 // Avoid log of non-positive numbers
		}
		return math.Log(left)
	}
	return 0
}

// Evaluate calculates the fitness of an individual
func (ind *Individual) Evaluate(x [][]float64, y []float64, complexityPenalty float64) {
	mse := 0.0
	for i := range y {
		predicted := Evaluate(ind.Tree, x[i])
		if math.IsNaN(predicted) || math.IsInf(predicted, 0) {
			ind.Fitness = math.Inf(1)
			ind.AdjustedFitness = math.Inf(1)
			return
		}
		mse += math.Pow(predicted-y[i], 2)
	}
	ind.Fitness = mse / float64(len(y))
	ind.Complexity = CalculateComplexity(ind.Tree)
	ind.AdjustedFitness = ind.Fitness + complexityPenalty*float64(ind.Complexity)
}

// CalculateComplexity returns the number of nodes in the tree
func CalculateComplexity(node *Node) int {
	if node == nil {
		return 0
	}
	return 1 + CalculateComplexity(node.Left) + CalculateComplexity(node.Right)
}

func CalculateR2(y []float64, predictions []float64) float64 {
	yMean := 0.0
	for _, yi := range y {
		yMean += yi
	}
	yMean /= float64(len(y))

	ssTotal := 0.0
	ssResidual := 0.0
	for i, yi := range y {
		ssTotal += math.Pow(yi-yMean, 2)
		ssResidual += math.Pow(yi-predictions[i], 2)
	}

	return 1 - (ssResidual / ssTotal)
}

func CalculateMAE(y []float64, predictions []float64) float64 {
	sum := 0.0
	for i := range y {
		sum += math.Abs(y[i] - predictions[i])
	}
	return sum / float64(len(y))
}

// CreateRandomTree generates a random expression tree
func CreateRandomTree(depth int, maxDepth int, varCount int) *Node {
	if depth >= maxDepth || (depth > 0 && rand.Float64() < 0.1) {
		if rand.Float64() < 0.5 {
			return &Node{Value: rand.Float64()*10 - 5, IsLeaf: true}
		}
		return &Node{Variable: rand.Intn(varCount), IsLeaf: true}
	}

	node := &Node{Op: rand.Intn(8)} // For the number of operations that we have.
	node.Left = CreateRandomTree(depth+1, maxDepth, varCount)
	if node.Op < LOG { // Binary operators
		node.Right = CreateRandomTree(depth+1, maxDepth, varCount)
	}
	return node
}

// Crossover performs crossover between two parent trees
func Crossover(parent1, parent2 *Node) (*Node, *Node) {
	if parent1 == nil || parent2 == nil {
		return CloneTree(parent1), CloneTree(parent2)
	}
	child1 := CloneTree(parent1)
	child2 := CloneTree(parent2)

	node1 := GetRandomNode(child1)
	node2 := GetRandomNode(child2)

	if node1 != nil && node2 != nil {
		temp := CloneTree(node1)
		ReplaceNode(child1, node1, CloneTree(node2))
		ReplaceNode(child2, node2, temp)
	}

	return child1, child2
}

// ReplaceNode replaces oldNode with newNode in the tree
func ReplaceNode(tree *Node, oldNode *Node, newNode *Node) {
	if tree == oldNode {
		*tree = *newNode
		return
	}
	if tree.Left == oldNode {
		tree.Left = newNode
		return
	}
	if tree.Right == oldNode {
		tree.Right = newNode
		return
	}
	if tree.Left != nil {
		ReplaceNode(tree.Left, oldNode, newNode)
	}
	if tree.Right != nil {
		ReplaceNode(tree.Right, oldNode, newNode)
	}
}

// Mutate performs mutation on a tree
func Mutate(tree *Node, varCount int) {
	if tree == nil {
		return
	}
	if rand.Float64() < 0.1 {
		node := GetRandomNode(tree)
		if node == nil {
			return
		}
		if node.IsLeaf {
			if rand.Float64() < 0.5 {
				node.IsLeaf = false
				node.Op = rand.Intn(8)
				node.Left = CreateRandomTree(0, 2, varCount)
				if node.Op < LOG {
					node.Right = CreateRandomTree(0, 2, varCount)
				}
			} else {
				if rand.Float64() < 0.5 {
					node.Variable = rand.Intn(varCount)
				} else {
					node.Value = rand.Float64()*10 - 5
				}
			}
		} else {
			node.Op = rand.Intn(8) // For the 8 operators...
			if node.Op >= LOG && node.Right != nil {
				node.Right = nil
			} else if node.Op < LOG && node.Right == nil {
				node.Right = CreateRandomTree(0, 2, varCount)
			}
		}
	}
}

// CloneTree creates a deep copy of a tree
func CloneTree(node *Node) *Node {
	if node == nil {
		return nil
	}
	newNode := &Node{
		Op:       node.Op,
		Value:    node.Value,
		IsLeaf:   node.IsLeaf,
		Variable: node.Variable,
	}
	newNode.Left = CloneTree(node.Left)
	newNode.Right = CloneTree(node.Right)
	return newNode
}

// GetRandomNode returns a random node from the tree
func GetRandomNode(node *Node) *Node {
	if node == nil {
		return nil
	}
	nodes := make([]*Node, 0)
	collectNodes(node, &nodes)
	if len(nodes) == 0 {
		return nil
	}
	return nodes[rand.Intn(len(nodes))]
}

func collectNodes(node *Node, nodes *[]*Node) {
	if node == nil {
		return
	}
	*nodes = append(*nodes, node)
	collectNodes(node.Left, nodes)
	collectNodes(node.Right, nodes)
}

// InitializePopulation creates an initial population
func InitializePopulation(size int, maxDepth int, varCount int) Population {
	population := make(Population, size)
	for i := range population {
		tree := CreateRandomTree(0, maxDepth, varCount)
		population[i] = &Individual{Tree: tree}
	}
	return population
}

// Selection performs tournament selection
func Selection(population Population, tournamentSize int) *Individual {
	tournament := make(Population, tournamentSize)
	for i := range tournament {
		tournament[i] = population[rand.Intn(len(population))]
	}
	sort.Slice(tournament, func(i, j int) bool {
		return tournament[i].AdjustedFitness < tournament[j].AdjustedFitness
	})
	return tournament[0]
}

// EvolvePopulation creates a new generation through selection, crossover, and mutation
func EvolvePopulation(population Population, eliteSize int, varCount int) Population {
	newPopulation := make(Population, len(population))
	sort.Slice(population, func(i, j int) bool {
		return population[i].AdjustedFitness < population[j].AdjustedFitness
	})

	// Elitism
	for i := 0; i < eliteSize; i++ {
		newPopulation[i] = &Individual{Tree: CloneTree(population[i].Tree)}
	}

	// Crossover and Mutation
	var wg sync.WaitGroup
	for i := eliteSize; i < len(population); i++ {
		wg.Add(1)
		go func(index int) {
			defer wg.Done()
			parent1 := Selection(population, 5)
			parent2 := Selection(population, 5)
			child1, _ := Crossover(parent1.Tree, parent2.Tree)
			Mutate(child1, varCount)
			newPopulation[index] = &Individual{Tree: child1}
		}(i)
	}
	wg.Wait()

	return newPopulation
}

// PrintTree prints the expression tree (infix notation)
func PrintTree(node *Node, varNames []string) string {
	if node == nil {
		return ""
	}
	if node.IsLeaf {
		if node.Variable >= 0 && node.Variable < len(varNames) {
			return varNames[node.Variable]
		}
		return fmt.Sprintf("%.2f", node.Value)
	}
	if node.Op == SIN || node.Op == COS || node.Op == LOG {
		return fmt.Sprintf("%s(%s)", OpToString(node.Op), PrintTree(node.Left, varNames))
	}
	if node.Op == POW {
		return fmt.Sprintf("(%s ^ %s)", PrintTree(node.Left, varNames), PrintTree(node.Right, varNames))
	}

	return fmt.Sprintf("(%s %s %s)", PrintTree(node.Left, varNames), OpToString(node.Op), PrintTree(node.Right, varNames))
}

func OpToString(op int) string {
	switch op {
	case ADD:
		return "+"
	case SUB:
		return "-"
	case MUL:
		return "*"
	case DIV:
		return "/"
	case SIN:
		return "sin"
	case COS:
		return "cos"

	case POW:
		return "^"

	case LOG:
		return "log"

	}
	return "?"
}

func readCSV(filepath string) ([][]string, error) {
	file, err := os.Open(filepath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	return records, nil
}

func parseCSVData(records [][]string, explanatoryVars []string, predictVar string) ([][]float64, []float64, error) {
	headers := records[0]
	xIndices := make([]int, len(explanatoryVars))
	for i, varName := range explanatoryVars {
		for j, header := range headers {
			if header == varName {
				xIndices[i] = j
				break
			}
		}
		if xIndices[i] == 0 && varName != headers[0] {
			return nil, nil, fmt.Errorf("explanatory variable %s not found in CSV", varName)
		}
	}

	yIndex := -1
	for i, header := range headers {
		if header == predictVar {
			yIndex = i
			break
		}
	}
	if yIndex == -1 {
		return nil, nil, fmt.Errorf("predict variable %s not found in CSV", predictVar)
	}

	x := make([][]float64, len(records)-1)
	y := make([]float64, len(records)-1)

	for i, record := range records[1:] {
		x[i] = make([]float64, len(explanatoryVars))
		for j, index := range xIndices {
			value, err := strconv.ParseFloat(record[index], 64)
			if err != nil {
				return nil, nil, fmt.Errorf("error parsing value in row %d, column %d: %v", i+2, index+1, err)
			}
			x[i][j] = value
		}

		yValue, err := strconv.ParseFloat(record[yIndex], 64)
		if err != nil {
			return nil, nil, fmt.Errorf("error parsing Y value in row %d: %v", i+2, err)
		}
		y[i] = yValue
	}

	return x, y, nil
}

func writeOutputCSV(x [][]float64, y []float64, predictions []float64, expression string) error {
	file, err := os.Create("data.csv")
	if err != nil {
		return err
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	headers := make([]string, len(x[0])+3)
	for i := range x[0] {
		headers[i] = fmt.Sprintf("X%d", i+1)
	}
	headers[len(x[0])] = "Y_Actual"
	headers[len(x[0])+1] = "Y_Predicted"
	headers[len(x[0])+2] = "Expression"

	writer.Write(headers)

	for i := range y {
		row := make([]string, len(x[0])+3)
		for j, val := range x[i] {
			row[j] = strconv.FormatFloat(val, 'f', 6, 64)
		}
		row[len(x[0])] = strconv.FormatFloat(y[i], 'f', 6, 64)
		row[len(x[0])+1] = strconv.FormatFloat(predictions[i], 'f', 6, 64)
		if i == 0 {
			row[len(x[0])+2] = expression
		}
		writer.Write(row)
	}

	return nil
}

func runWithCSV() {
	fmt.Println("Enter the path of the CSV file:")
	reader := bufio.NewReader(os.Stdin)
	filepath, _ := reader.ReadString('\n')
	filepath = strings.TrimSpace(filepath)

	records, err := readCSV(filepath)
	if err != nil {
		fmt.Printf("Error reading CSV: %v\n", err)
		return
	}

	fmt.Println("Enter the names of the explanatory variables (comma-separated if multiple, e.g., x1,x2):")
	explanatoryVarsInput, _ := reader.ReadString('\n')
	explanatoryVarsInput = strings.TrimSpace(explanatoryVarsInput)
	explanatoryVars := strings.Split(explanatoryVarsInput, ",")

	fmt.Println("Enter the name of the column with the variable to predict:")
	predictVar, _ := reader.ReadString('\n')
	predictVar = strings.TrimSpace(predictVar)

	x, y, err := parseCSVData(records, explanatoryVars, predictVar)
	if err != nil {
		fmt.Printf("Error parsing CSV data: %v\n", err)
		return
	}

	// Run genetic algorithm
	varCount := len(explanatoryVars)
	populationSize := 1000
	maxGenerations := 100
	eliteSize := 10
	maxDepth := 5
	complexityPenalty := 0.01

	population := InitializePopulation(populationSize, maxDepth, varCount)

	var bestSolution *Individual
	for generation := 0; generation < maxGenerations; generation++ {
		var wg sync.WaitGroup
		for _, ind := range population {
			wg.Add(1)
			go func(individual *Individual) {
				defer wg.Done()
				individual.Evaluate(x, y, complexityPenalty)
			}(ind)
		}
		wg.Wait()

		sort.Slice(population, func(i, j int) bool {
			return population[i].AdjustedFitness < population[j].AdjustedFitness
		})

		if bestSolution == nil || population[0].AdjustedFitness < bestSolution.AdjustedFitness {
			bestSolution = &Individual{Tree: CloneTree(population[0].Tree), Fitness: population[0].Fitness, Complexity: population[0].Complexity, AdjustedFitness: population[0].AdjustedFitness}
		}

		if generation%10 == 0 || generation == maxGenerations-1 {
			fmt.Printf("Generation %d: Best Fitness (MSE) = %f, Complexity = %d, Adjusted Fitness = %f\n",
				generation, population[0].Fitness, population[0].Complexity, population[0].AdjustedFitness)
		}

		population = EvolvePopulation(population, eliteSize, varCount)
	}

	fmt.Printf("Best solution: Fitness (MSE) = %f, Complexity = %d, Adjusted Fitness = %f\n",
		bestSolution.Fitness, bestSolution.Complexity, bestSolution.AdjustedFitness)
	fmt.Printf("Found expression: %s\n", PrintTree(bestSolution.Tree, explanatoryVars))

	// Calculate predictions, R2, and MAE
	predictions := make([]float64, len(y))
	for i := range y {
		predictions[i] = Evaluate(bestSolution.Tree, x[i])
	}

	r2 := CalculateR2(y, predictions)
	mae := CalculateMAE(y, predictions)
	rmse := math.Sqrt(bestSolution.Fitness)

	fmt.Printf("R-squared (R2) = %f\n", r2)
	fmt.Printf("Mean Absolute Error (MAE) = %f\n", mae)
	fmt.Printf("Root Mean Squared Error (RMSE) = %f\n", rmse)

	fmt.Println("Do you want to write the results to a CSV? (y/n)")
	writeCSVChoice, _ := reader.ReadString('\n')
	writeCSVChoice = strings.TrimSpace(writeCSVChoice)

	if writeCSVChoice == "y" || writeCSVChoice == "Y" {
		err := writeOutputCSV(x, y, predictions, PrintTree(bestSolution.Tree, explanatoryVars))
		if err != nil {
			fmt.Printf("Error writing to CSV: %v\n", err)
		} else {
			fmt.Println("Results written to data.csv")

		}
	}
}

func runExamples() {
	rand.Seed(time.Now().UnixNano())

	testCases := []struct {
		name           string
		dataGenerator  func(size int) ([][]float64, []float64)
		varNames       []string
		populationSize int
		maxGenerations int
		eliteSize      int
		maxDepth       int
		dataSize       int
	}{
		{
			name: "Linear Function",
			dataGenerator: func(size int) ([][]float64, []float64) {
				x := make([][]float64, size)
				y := make([]float64, size)
				for i := range y {
					x[i] = make([]float64, 1)
					x[i][0] = rand.Float64()*10 - 5
					y[i] = 2*x[i][0] + 1
				}
				return x, y
			},
			varNames:       []string{"x"},
			populationSize: 500,
			maxGenerations: 50,
			eliteSize:      5,
			maxDepth:       4,
			dataSize:       1000,
		},
		{
			name: "Quadratic Function",
			dataGenerator: func(size int) ([][]float64, []float64) {
				x := make([][]float64, size)
				y := make([]float64, size)
				for i := range y {
					x[i] = make([]float64, 1)
					x[i][0] = rand.Float64()*10 - 5
					y[i] = x[i][0]*x[i][0] + 2*x[i][0] + 1
				}
				return x, y
			},
			varNames:       []string{"x"},
			populationSize: 1000,
			maxGenerations: 100,
			eliteSize:      10,
			maxDepth:       5,
			dataSize:       2000,
		},
		{
			name: "Multiple Variables Linear",
			dataGenerator: func(size int) ([][]float64, []float64) {
				x := make([][]float64, size)
				y := make([]float64, size)
				for i := range y {
					x[i] = make([]float64, 3)
					x[i][0] = rand.Float64()*10 - 5
					x[i][1] = rand.Float64()*10 - 5
					x[i][2] = rand.Float64()*10 - 5
					y[i] = 2*x[i][0] - 3*x[i][1] + 0.5*x[i][2] + 1
				}
				return x, y
			},
			varNames:       []string{"x", "y", "z"},
			populationSize: 1500,
			maxGenerations: 150,
			eliteSize:      15,
			maxDepth:       6,
			dataSize:       3000,
		},
		{
			name: "Multiple Variables Nonlinear",
			dataGenerator: func(size int) ([][]float64, []float64) {
				x := make([][]float64, size)
				y := make([]float64, size)
				for i := range y {
					x[i] = make([]float64, 3)
					x[i][0] = rand.Float64()*10 - 5
					x[i][1] = rand.Float64()*10 - 5
					x[i][2] = rand.Float64()*10 - 5
					y[i] = math.Sin(x[i][0]*x[i][1]) + math.Log(math.Abs(x[i][2])+1) - math.Sqrt(math.Abs(x[i][0]*x[i][1]*x[i][2]))
				}
				return x, y
			},
			varNames:       []string{"x", "y", "z"},
			populationSize: 2000,
			maxGenerations: 200,
			eliteSize:      20,
			maxDepth:       8,
			dataSize:       4000,
		},
	}

	for _, tc := range testCases {
		fmt.Printf("\nRunning test case: %s\n", tc.name)
		fmt.Printf("Variables: %v\n", tc.varNames)

		x, y := tc.dataGenerator(tc.dataSize)
		varCount := len(tc.varNames)
		complexityPenalty := 0.01

		population := InitializePopulation(tc.populationSize, tc.maxDepth, varCount)

		var bestSolution *Individual
		for generation := 0; generation < tc.maxGenerations; generation++ {
			var wg sync.WaitGroup
			for _, ind := range population {
				wg.Add(1)
				go func(individual *Individual) {
					defer wg.Done()
					individual.Evaluate(x, y, complexityPenalty)
				}(ind)
			}
			wg.Wait()

			sort.Slice(population, func(i, j int) bool {
				return population[i].AdjustedFitness < population[j].AdjustedFitness
			})

			if bestSolution == nil || population[0].AdjustedFitness < bestSolution.AdjustedFitness {
				bestSolution = &Individual{Tree: CloneTree(population[0].Tree), Fitness: population[0].Fitness, Complexity: population[0].Complexity, AdjustedFitness: population[0].AdjustedFitness}
			}

			if generation%10 == 0 || generation == tc.maxGenerations-1 {
				fmt.Printf("Generation %d: Best Fitness (MSE) = %f, Complexity = %d, Adjusted Fitness = %f\n",
					generation, population[0].Fitness, population[0].Complexity, population[0].AdjustedFitness)
			}

			population = EvolvePopulation(population, tc.eliteSize, varCount)
		}

		fmt.Printf("Best solution: Fitness (MSE) = %f, Complexity = %d, Adjusted Fitness = %f\n",
			bestSolution.Fitness, bestSolution.Complexity, bestSolution.AdjustedFitness)
		fmt.Printf("Found expression: %s\n", PrintTree(bestSolution.Tree, tc.varNames))

		// Calculate predictions, R2, and MAE
		predictions := make([]float64, len(y))
		for i := range y {
			predictions[i] = Evaluate(bestSolution.Tree, x[i])
		}

		r2 := CalculateR2(y, predictions)
		mae := CalculateMAE(y, predictions)
		rmse := math.Sqrt(bestSolution.Fitness)

		fmt.Printf("R-squared (R2) = %f\n", r2)
		fmt.Printf("Mean Absolute Error (MAE) = %f\n", mae)
		fmt.Printf("Root Mean Squared Error (RMSE) = %f\n", rmse)

		// Perform a simple validation on a new dataset
		validationSize := tc.dataSize / 5 // 20% of original data size
		xValidation, yValidation := tc.dataGenerator(validationSize)

		validationMSE := 0.0
		for i := range yValidation {
			predicted := Evaluate(bestSolution.Tree, xValidation[i])
			validationMSE += math.Pow(predicted-yValidation[i], 2)
		}
		validationMSE /= float64(len(yValidation))

		fmt.Printf("Validation MSE = %f\n", validationMSE)

		fmt.Println("\nPress Enter to continue to the next test case...")
		fmt.Scanln() // Wait for user input before proceeding to the next test case
	}
}

func main() {
	fmt.Println("Welcome to the Symbolic Regression program!")
	fmt.Println("Press 1 to run examples")
	fmt.Println("Press 2 to run with a CSV")

	reader := bufio.NewReader(os.Stdin)
	choice, _ := reader.ReadString('\n')
	choice = strings.TrimSpace(choice)

	if choice == "1" {
		runExamples()
	} else if choice == "2" {
		runWithCSV()
	} else {
		fmt.Println("Invalid choice. Exiting.")
	}
}
