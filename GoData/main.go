package main

import (
	"encoding/csv"
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
)

type Series struct {
	data  []string
	dtype string
}

func NewSeries(data []string) *Series {
	return &Series{
		data:  data,
		dtype: inferDataType(data),
	}
}

func inferDataType(data []string) string {
	if len(data) == 0 {
		return "string"
	}

	isInt := true
	isFloat := true
	isBool := true

	for _, v := range data {
		if v == "" {
			continue
		}
		if _, err := strconv.Atoi(v); err != nil {
			isInt = false
		}
		if _, err := strconv.ParseFloat(v, 64); err != nil {
			isFloat = false
		}
		if _, err := strconv.ParseBool(v); err != nil {
			isBool = false
		}
		if !isInt && !isFloat && !isBool {
			return "string"
		}
	}

	if isInt {
		return "int"
	}
	if isFloat {
		return "float"
	}
	if isBool {
		return "bool"
	}
	return "string"
}

func (s *Series) toArray() []string {
	return s.data
}

func (s *Series) ToTypedArray() (interface{}, error) {
	switch s.dtype {
	case "int":
		intArray := make([]int, len(s.data))
		for i, v := range s.data {
			if v == "" {
				continue
			}
			intVal, err := strconv.Atoi(v)
			if err != nil {
				return nil, fmt.Errorf("error converting value to int: %v", err)
			}
			intArray[i] = intVal
		}
		return intArray, nil
	case "float":
		floatArray := make([]float64, len(s.data))
		for i, v := range s.data {
			if v == "" {
				continue
			}
			floatVal, err := strconv.ParseFloat(v, 64)
			if err != nil {
				return nil, fmt.Errorf("error converting value to float: %v", err)
			}
			floatArray[i] = floatVal
		}
		return floatArray, nil
	case "bool":
		boolArray := make([]bool, len(s.data))
		for i, v := range s.data {
			if v == "" {
				continue
			}
			boolVal, err := strconv.ParseBool(v)
			if err != nil {
				return nil, fmt.Errorf("error converting value to bool: %v", err)
			}
			boolArray[i] = boolVal
		}
		return boolArray, nil
	default:
		return s.data, nil
	}
}

func (s *Series) Unique() *Series {
	uniqueMap := make(map[string]bool)
	var uniqueData []string
	for _, item := range s.data {
		if _, exists := uniqueMap[item]; !exists {
			uniqueMap[item] = true
			uniqueData = append(uniqueData, item)
		}
	}
	return NewSeries(uniqueData)
}

func (s *Series) String() string {
	return "[" + strings.Join(s.data, ", ") + "]"
}

func (s *Series) asType(typeName string) (*Series, error) {
	newData := make([]string, len(s.data))

	for i, value := range s.data {
		var newValue string

		switch typeName {
		case "int":
			if intVal, err := strconv.Atoi(value); err == nil {
				newValue = strconv.Itoa(intVal)
			} else {
				return nil, fmt.Errorf("cannot convert value '%s' to int", value)
			}
		case "float":
			if floatVal, err := strconv.ParseFloat(value, 64); err == nil {
				newValue = strconv.FormatFloat(floatVal, 'f', -1, 64)
			} else {
				return nil, fmt.Errorf("cannot convert value '%s' to float", value)
			}
		case "string":
			newValue = value
		case "bool":
			if boolVal, err := strconv.ParseBool(value); err == nil {
				newValue = strconv.FormatBool(boolVal)
			} else {
				return nil, fmt.Errorf("cannot convert value '%s' to bool", value)
			}
		default:
			return nil, fmt.Errorf("unsupported type: %s", typeName)
		}

		newData[i] = newValue
	}

	return &Series{data: newData, dtype: typeName}, nil
}

type DataFrame struct {
	Columns   []string
	Data      [][]string
	ColumnMap map[string]*Series
}

type FilterCondition struct {
	Column   string
	Operator string
	Value    interface{}
}

func ReadCSVToDataFrame(filename string) (*DataFrame, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	data, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	return NewDataFrame(data), nil
}

func NewDataFrame(data [][]string) *DataFrame {
	df := &DataFrame{
		Columns:   data[0],
		Data:      data[1:],
		ColumnMap: make(map[string]*Series),
	}

	for i, colName := range df.Columns {
		column := make([]string, len(df.Data))
		for j, row := range df.Data {
			column[j] = row[i]
		}
		df.ColumnMap[colName] = NewSeries(column)
	}

	return df
}

func (df *DataFrame) Print() {
	fmt.Println("\n")
	colWidths := make([]int, len(df.Columns))
	for i, col := range df.Columns {
		colWidths[i] = len(col)
	}
	for _, row := range df.Data {
		for i, cell := range row {
			if len(cell) > colWidths[i] {
				colWidths[i] = len(cell)
			}
		}
	}

	for i, col := range df.Columns {
		fmt.Printf("%-*s\t", colWidths[i], col)
	}
	fmt.Println()

	for _, width := range colWidths {
		fmt.Print(strings.Repeat("-", width) + "\t")
	}
	fmt.Println()

	for _, row := range df.Data {
		for i, cell := range row {
			fmt.Printf("%-*s\t", colWidths[i], cell)
		}
		fmt.Println()
	}
}

func evaluateConditions(df *DataFrame, row []string, conditions []FilterCondition) bool {
	for _, condition := range conditions {
		if !evaluateCondition(df, row, condition) {
			return false
		}
	}
	return true
}

func evaluateCondition(df *DataFrame, row []string, condition FilterCondition) bool {
	colIndex := df.getColumnIndex(condition.Column)
	if colIndex == -1 {
		return false
	}

	value := row[colIndex]
	switch condition.Operator {
	case "==":
		return value == fmt.Sprintf("%v", condition.Value)
	case "!=":
		return value != fmt.Sprintf("%v", condition.Value)
	case ">", "<", ">=", "<=":
		v1, err1 := strconv.ParseFloat(value, 64)
		v2, err2 := strconv.ParseFloat(fmt.Sprintf("%v", condition.Value), 64)
		if err1 != nil || err2 != nil {
			return false
		}
		switch condition.Operator {
		case ">":
			return v1 > v2
		case "<":
			return v1 < v2
		case ">=":
			return v1 >= v2
		case "<=":
			return v1 <= v2
		}
	}
	return false
}

func (df *DataFrame) Col(name string) *Series {
	return df.ColumnMap[name]
}

func (df *DataFrame) getColumnIndex(name string) int {
	for i, colName := range df.Columns {
		if colName == name {
			return i
		}
	}
	return -1
}

func (df *DataFrame) Length() int {
	return len(df.Data)
}

func (df *DataFrame) RemoveDuplicates(subset []string) *DataFrame {
	if len(subset) == 0 {
		subset = df.Columns
	}

	seen := make(map[string]bool)
	var newData [][]string

	for _, row := range df.Data {
		key := make([]string, len(subset))
		for i, col := range subset {
			key[i] = row[df.getColumnIndex(col)]
		}
		keyStr := strings.Join(key, "|")
		if !seen[keyStr] {
			seen[keyStr] = true
			newData = append(newData, row)
		}
	}

	return NewDataFrame(append([][]string{df.Columns}, newData...))
}

func (df *DataFrame) AddColumn(name string, data interface{}) error {
	var newColumn []string

	switch v := data.(type) {
	case func(row []string) string:
		newColumn = make([]string, len(df.Data))
		for i, row := range df.Data {
			newColumn[i] = v(row)
		}
	case []string:
		if len(v) != len(df.Data) {
			return fmt.Errorf("array length (%d) does not match DataFrame length (%d)", len(v), len(df.Data))
		}
		newColumn = v
	case []int:
		if len(v) != len(df.Data) {
			return fmt.Errorf("array length (%d) does not match DataFrame length (%d)", len(v), len(df.Data))
		}
		newColumn = make([]string, len(v))
		for i, val := range v {
			newColumn[i] = strconv.Itoa(val)
		}
	case []float64:
		if len(v) != len(df.Data) {
			return fmt.Errorf("array length (%d) does not match DataFrame length (%d)", len(v), len(df.Data))
		}
		newColumn = make([]string, len(v))
		for i, val := range v {
			newColumn[i] = strconv.FormatFloat(val, 'f', -1, 64)
		}
	default:
		return fmt.Errorf("unsupported data type for AddColumn")
	}

	if existingCol, exists := df.ColumnMap[name]; exists {
		// Update existing column
		existingCol.data = newColumn
		for i, row := range df.Data {
			colIndex := df.getColumnIndex(name)
			row[colIndex] = newColumn[i]
		}
	} else {
		// Add new column
		df.Columns = append(df.Columns, name)
		df.ColumnMap[name] = NewSeries(newColumn)
		for i, row := range df.Data {
			df.Data[i] = append(row, newColumn[i])
		}
	}

	return nil
}

// Add this method to your DataFrame struct
func (df *DataFrame) UpdateColumn(name string, updateFunc func(value string) string) error {
	colIndex := df.getColumnIndex(name)
	if colIndex == -1 {
		return fmt.Errorf("column '%s' not found", name)
	}

	for i, row := range df.Data {
		df.Data[i][colIndex] = updateFunc(row[colIndex])
	}

	df.ColumnMap[name].data = df.Col(name).data

	return nil
}

func (df *DataFrame) Where(condition interface{}, trueValue, falseValue interface{}) *Series {
	result := make([]string, len(df.Data))

	for i, row := range df.Data {
		var conditionMet bool
		switch cond := condition.(type) {
		case string:
			conditionMet = evaluateStringCondition(df, row, cond)
		case []FilterCondition:
			conditionMet = evaluateFilterConditions(df, row, cond)
		default:
			panic("Unsupported condition type")
		}

		if conditionMet {
			result[i] = applyValue(trueValue, row)
		} else {
			result[i] = applyValue(falseValue, row)
		}
	}

	return NewSeries(result)
}

// Add this function to evaluate FilterCondition slices
func evaluateFilterConditions(df *DataFrame, row []string, conditions []FilterCondition) bool {
	for _, condition := range conditions {
		if !evaluateCondition(df, row, condition) {
			return false
		}
	}
	return true
}

func (df *DataFrame) Filter(condition interface{}) *DataFrame {
	var filteredData [][]string

	switch cond := condition.(type) {
	case string:
		for _, row := range df.Data {
			if evaluateStringCondition(df, row, cond) {
				filteredData = append(filteredData, row)
			}
		}
	case []FilterCondition:
		for _, row := range df.Data {
			if evaluateFilterConditions(df, row, cond) {
				filteredData = append(filteredData, row)
			}
		}
	default:
		panic("Unsupported condition type")
	}

	// If no rows match the condition, return an empty DataFrame with the same columns
	if len(filteredData) == 0 {
		return NewDataFrame([][]string{df.Columns})
	}

	return NewDataFrame(append([][]string{df.Columns}, filteredData...))
}

// applyValue function
func applyValue(value interface{}, row []string) string {
	switch v := value.(type) {
	case string:
		return v
	case func([]string) string:
		return v(row)
	case int:
		return strconv.Itoa(v)
	case float64:
		return strconv.FormatFloat(v, 'f', -1, 64)
	case bool:
		return strconv.FormatBool(v)
	default:
		return fmt.Sprintf("%v", v)
	}
}

func (df *DataFrame) GroupBy(categoricalCols []string, numericalCols []string, aggregations []string) (*DataFrame, error) {
	groups := make(map[string][][]string)

	for _, row := range df.Data {
		key := make([]string, len(categoricalCols))
		for i, col := range categoricalCols {
			colIndex := df.getColumnIndex(col)
			if colIndex == -1 {
				return nil, fmt.Errorf("categorical column not found: %s", col)
			}
			key[i] = row[colIndex]
		}
		keyStr := strings.Join(key, "|")
		groups[keyStr] = append(groups[keyStr], row)
	}

	var newData [][]string
	newColumns := append(categoricalCols, make([]string, 0, len(numericalCols)*len(aggregations))...)
	for _, numCol := range numericalCols {
		for _, agg := range aggregations {
			newColumns = append(newColumns, fmt.Sprintf("%s_%s", numCol, agg))
		}
	}

	for _, rows := range groups {
		newRow := make([]string, len(newColumns))
		for i, col := range categoricalCols {
			colIndex := df.getColumnIndex(col)
			if colIndex == -1 {
				return nil, fmt.Errorf("categorical column not found: %s", col)
			}
			newRow[i] = rows[0][colIndex]
		}

		i := len(categoricalCols)
		for _, numCol := range numericalCols {
			colIndex := df.getColumnIndex(numCol)
			if colIndex == -1 {
				return nil, fmt.Errorf("numerical column not found: %s", numCol)
			}
			values := make([]float64, len(rows))
			for j, row := range rows {
				val, err := strconv.ParseFloat(row[colIndex], 64)
				if err != nil {
					return nil, fmt.Errorf("invalid numeric value in column %s: %s", numCol, row[colIndex])
				}
				values[j] = val
			}

			for _, agg := range aggregations {
				var result float64
				switch agg {
				case "sum":
					result = calculateSum(values)
				case "min":
					result = calculateMin(values)
				case "max":
					result = calculateMax(values)
				case "avg":
					result = calculateMean(values)
				case "std":
					result = calculateStdDev(values, calculateMean(values))
				default:
					return nil, fmt.Errorf("unsupported aggregation function: %s", agg)
				}
				newRow[i] = fmt.Sprintf("%.2f", result)
				i++
			}
		}

		newData = append(newData, newRow)
	}

	return NewDataFrame(append([][]string{newColumns}, newData...)), nil
}

func (df *DataFrame) Join(other *DataFrame, joinType string, leftOn, rightOn string) (*DataFrame, error) {
	leftIndex := df.getColumnIndex(leftOn)
	rightIndex := other.getColumnIndex(rightOn)
	if leftIndex == -1 || rightIndex == -1 {
		return nil, fmt.Errorf("join columns not found")
	}

	leftMap := make(map[string][]int)
	for i, row := range df.Data {
		key := row[leftIndex]
		leftMap[key] = append(leftMap[key], i)
	}

	var newColumns []string
	newColumns = append(newColumns, df.Columns...)
	for _, col := range other.Columns {
		if col != rightOn {
			newColumns = append(newColumns, col)
		}
	}

	var newData [][]string
	switch joinType {
	case "inner":
		for _, rightRow := range other.Data {
			rightKey := rightRow[rightIndex]
			if leftIndices, ok := leftMap[rightKey]; ok {
				for _, leftIndex := range leftIndices {
					newRow := append([]string{}, df.Data[leftIndex]...)
					for i, val := range rightRow {
						if i != rightIndex {
							newRow = append(newRow, val)
						}
					}
					newData = append(newData, newRow)
				}
			}
		}
	case "left":
		for i, leftRow := range df.Data {
			leftKey := leftRow[leftIndex]
			if rightIndices, ok := leftMap[leftKey]; ok {
				for _, rightIndex := range rightIndices {
					newRow := append([]string{}, leftRow...)
					for i, val := range other.Data[rightIndex] {
						if i != rightIndex {
							newRow = append(newRow, val)
						}
					}
					newData = append(newData, newRow)
				}
			} else {
				newRow := append([]string{}, leftRow...)
				for range other.Columns {
					if i != rightIndex {
						newRow = append(newRow, "")
					}
				}
				newData = append(newData, newRow)
			}
		}
	case "outer":
		rightMap := make(map[string]bool)
		for i, leftRow := range df.Data {
			leftKey := leftRow[leftIndex]
			if rightIndices, ok := leftMap[leftKey]; ok {
				for _, rightIndex := range rightIndices {
					newRow := append([]string{}, leftRow...)
					for i, val := range other.Data[rightIndex] {
						if i != rightIndex {
							newRow = append(newRow, val)
						}
					}
					newData = append(newData, newRow)
				}
				rightMap[leftKey] = true
			} else {
				newRow := append([]string{}, leftRow...)
				for range other.Columns {
					if i != rightIndex {
						newRow = append(newRow, "")
					}
				}
				newData = append(newData, newRow)
			}
		}
		for _, rightRow := range other.Data {
			rightKey := rightRow[rightIndex]
			if !rightMap[rightKey] {
				newRow := make([]string, len(df.Columns))
				for i := range newRow {
					newRow[i] = ""
				}
				newRow[leftIndex] = rightKey
				for i, val := range rightRow {
					if i != rightIndex {
						newRow = append(newRow, val)
					}
				}
				newData = append(newData, newRow)
			}
		}
	default:
		return nil, fmt.Errorf("unsupported join type: %s", joinType)
	}

	return NewDataFrame(append([][]string{newColumns}, newData...)), nil
}

func (df *DataFrame) Sort(columns []string, ascending []bool) (*DataFrame, error) {
	if len(columns) != len(ascending) {
		return nil, fmt.Errorf("length of columns and ascending must match")
	}

	sortedDF := NewDataFrame(append([][]string{df.Columns}, df.Data...))

	sort.SliceStable(sortedDF.Data, func(i, j int) bool {
		for k, col := range columns {
			colIndex := sortedDF.getColumnIndex(col)
			if colIndex == -1 {
				return false
			}

			cmp := strings.Compare(sortedDF.Data[i][colIndex], sortedDF.Data[j][colIndex])
			if cmp != 0 {
				if ascending[k] {
					return cmp < 0
				} else {
					return cmp > 0
				}
			}
		}
		return false
	})

	return sortedDF, nil
}

func (df *DataFrame) AsType(column string, typeName string) error {
	colIndex := df.getColumnIndex(column)
	if colIndex == -1 {
		return fmt.Errorf("column not found: %s", column)
	}

	series := df.Col(column)
	newSeries, err := series.asType(typeName)
	if err != nil {
		return err
	}

	df.ColumnMap[column] = newSeries
	for i, row := range df.Data {
		row[colIndex] = newSeries.data[i]
	}

	return nil
}

func (df *DataFrame) Head(n int) *DataFrame {
	if n > len(df.Data) {
		n = len(df.Data)
	}
	return NewDataFrame(append([][]string{df.Columns}, df.Data[:n]...))
}

func (df *DataFrame) Tail(n int) *DataFrame {
	if n > len(df.Data) {
		n = len(df.Data)
	}
	return NewDataFrame(append([][]string{df.Columns}, df.Data[len(df.Data)-n:]...))
}
func (df *DataFrame) Describe() *DataFrame {
	stats := []string{"count", "mean", "std", "min", "25%", "50%", "75%", "max"}
	newData := make([][]string, len(stats))
	for i := range newData {
		newData[i] = make([]string, len(df.Columns)+1)
		newData[i][0] = stats[i]
	}

	for j, col := range df.Columns {
		series := df.Col(col)
		floats, err := series.ToFloat64()
		if err != nil {
			newData[0][j+1] = fmt.Sprintf("%d", len(series.data)) // count
			for i := 1; i < len(stats); i++ {
				newData[i][j+1] = "NaN"
			}
			continue
		}

		count := float64(len(floats))
		mean := calculateMean(floats)
		std := calculateStdDev(floats, mean)
		min := calculateMin(floats)
		max := calculateMax(floats)

		sortedFloats := make([]float64, len(floats))
		copy(sortedFloats, floats)
		sort.Float64s(sortedFloats)

		q1 := calculatePercentile(sortedFloats, 0.25)
		median := calculatePercentile(sortedFloats, 0.5)
		q3 := calculatePercentile(sortedFloats, 0.75)

		newData[0][j+1] = fmt.Sprintf("%.0f", count)
		newData[1][j+1] = fmt.Sprintf("%.2f", mean)
		newData[2][j+1] = fmt.Sprintf("%.2f", std)
		newData[3][j+1] = fmt.Sprintf("%.2f", min)
		newData[4][j+1] = fmt.Sprintf("%.2f", q1)
		newData[5][j+1] = fmt.Sprintf("%.2f", median)
		newData[6][j+1] = fmt.Sprintf("%.2f", q3)
		newData[7][j+1] = fmt.Sprintf("%.2f", max)
	}

	return NewDataFrame(append([][]string{append([]string{""}, df.Columns...)}, newData...))
}

func (s *Series) ToFloat64() ([]float64, error) {
	floats := make([]float64, len(s.data))
	for i, v := range s.data {
		if v == "" {
			continue // Skip empty values
		}
		f, err := strconv.ParseFloat(v, 64)
		if err != nil {
			return nil, fmt.Errorf("error converting value to float64: %v", err)
		}
		floats[i] = f
	}
	return floats, nil
}

func (df *DataFrame) Select(columns []string) *DataFrame {
	newColumns := make([]string, 0, len(columns))
	columnIndices := make([]int, 0, len(columns))

	for _, col := range columns {
		index := df.getColumnIndex(col)
		if index != -1 {
			newColumns = append(newColumns, col)
			columnIndices = append(columnIndices, index)
		}
	}

	newData := make([][]string, len(df.Data))
	for i, row := range df.Data {
		newRow := make([]string, len(columnIndices))
		for j, index := range columnIndices {
			newRow[j] = row[index]
		}
		newData[i] = newRow
	}

	return NewDataFrame(append([][]string{newColumns}, newData...))
}
func (df *DataFrame) Drop(columns ...interface{}) *DataFrame {
	var dropColumns []string

	// Handle both individual strings and []string
	for _, col := range columns {
		switch v := col.(type) {
		case string:
			dropColumns = append(dropColumns, v)
		case []string:
			dropColumns = append(dropColumns, v...)
		default:
			panic("Unsupported type for Drop method")
		}
	}

	newColumns := make([]string, 0, len(df.Columns))
	columnIndices := make([]int, 0, len(df.Columns))

	// Create a map for quick lookup of columns to drop
	dropMap := make(map[string]bool)
	for _, col := range dropColumns {
		dropMap[col] = true
	}

	for i, col := range df.Columns {
		if !dropMap[col] {
			newColumns = append(newColumns, col)
			columnIndices = append(columnIndices, i)
		}
	}

	newData := make([][]string, len(df.Data))
	for i, row := range df.Data {
		newRow := make([]string, len(columnIndices))
		for j, index := range columnIndices {
			newRow[j] = row[index]
		}
		newData[i] = newRow
	}

	newDF := NewDataFrame(append([][]string{newColumns}, newData...))
	return newDF
}

func (df *DataFrame) DropNA() *DataFrame {
	var newData [][]string
	for _, row := range df.Data {
		hasNA := false
		for _, cell := range row {
			if cell == "" || cell == "NA" || cell == "NaN" {
				hasNA = true
				break
			}
		}
		if !hasNA {
			newData = append(newData, row)
		}
	}
	return NewDataFrame(append([][]string{df.Columns}, newData...))
}

func (df *DataFrame) FillNA(value string) *DataFrame {
	newData := make([][]string, len(df.Data))
	for i, row := range df.Data {
		newRow := make([]string, len(row))
		for j, cell := range row {
			if cell == "" || cell == "NA" || cell == "NaN" {
				newRow[j] = value
			} else {
				newRow[j] = cell
			}
		}
		newData[i] = newRow
	}
	return NewDataFrame(append([][]string{df.Columns}, newData...))
}

func (df *DataFrame) Merge(other *DataFrame, on string, how string) (*DataFrame, error) {
	return df.Join(other, how, on, on)
}

func (df *DataFrame) ApplyFunc(column string, f func(string) string) *DataFrame {
	newData := make([][]string, len(df.Data))
	colIndex := df.getColumnIndex(column)
	if colIndex == -1 {
		return df
	}

	for i, row := range df.Data {
		newRow := make([]string, len(row))
		copy(newRow, row)
		newRow[colIndex] = f(row[colIndex])
		newData[i] = newRow
	}

	return NewDataFrame(append([][]string{df.Columns}, newData...))
}

func calculateSum(data []float64) float64 {
	sum := 0.0
	for _, v := range data {
		sum += v
	}
	return sum
}

func calculateMean(data []float64) float64 {
	return calculateSum(data) / float64(len(data))
}

func calculateStdDev(data []float64, mean float64) float64 {
	sumSquaredDiff := 0.0
	for _, v := range data {
		diff := v - mean
		sumSquaredDiff += diff * diff
	}
	variance := sumSquaredDiff / float64(len(data))
	return math.Sqrt(variance)
}

func calculateMin(data []float64) float64 {
	min := data[0]
	for _, v := range data[1:] {
		if v < min {
			min = v
		}
	}
	return min
}

func calculateMax(data []float64) float64 {
	max := data[0]
	for _, v := range data[1:] {
		if v > max {
			max = v
		}
	}
	return max
}

func calculatePercentile(sortedData []float64, percentile float64) float64 {
	index := percentile * float64(len(sortedData)-1)
	lower := math.Floor(index)
	upper := math.Ceil(index)
	if lower == upper {
		return sortedData[int(lower)]
	}
	return sortedData[int(lower)]*(upper-index) + sortedData[int(upper)]*(index-lower)

}

func evaluateStringCondition(df *DataFrame, row []string, condition string) bool {
	condition = strings.TrimSpace(condition)
	if strings.HasPrefix(condition, "(") && strings.HasSuffix(condition, ")") {
		condition = condition[1 : len(condition)-1]
	}

	orParts := strings.Split(condition, " OR ")

	for _, orPart := range orParts {
		andConditions := strings.Split(orPart, " AND ")
		allTrue := true

		for _, andCondition := range andConditions {
			andCondition = strings.TrimSpace(andCondition)
			if !evaluateSimpleCondition(df, row, andCondition) {
				allTrue = false
				break
			}
		}

		if allTrue {
			return true
		}
	}

	return false
}
func evaluateSimpleCondition(df *DataFrame, row []string, condition string) bool {
	parts := strings.Fields(condition)
	if len(parts) != 3 {
		return false
	}

	column, operator, value := parts[0], parts[1], parts[2]
	colIndex := df.getColumnIndex(column)
	if colIndex == -1 {
		return false
	}

	cellValue := row[colIndex]

	switch operator {
	case "==":
		return cellValue == value
	case "!=":
		return cellValue != value
	case ">", "<", ">=", "<=":
		cellFloat, err1 := strconv.ParseFloat(cellValue, 64)
		valueFloat, err2 := strconv.ParseFloat(value, 64)
		if err1 != nil || err2 != nil {
			return false
		}
		switch operator {
		case ">":
			return cellFloat > valueFloat
		case "<":
			return cellFloat < valueFloat
		case ">=":
			return cellFloat >= valueFloat
		case "<=":
			return cellFloat <= valueFloat
		}
	}

	return false
}

func main() {
	// Example 1: Reading CSV and basic operations
  // NOTE: this only works based on the provided data.csv file.
	df1, err := ReadCSVToDataFrame("data.csv")
	if err != nil {
		fmt.Println("Error reading CSV:", err)
		return
	}

	fmt.Println("DataFrame 1:")
	df1.Head(10).Print()

	// Example 2: Filtering using string condition
	stringCondition := "Purchase > 30"
	filteredDF := df1.Filter(stringCondition)
	fmt.Println("\nFiltered DataFrame (Purchase > 30):")
	filteredDF.Head(10).Print()

	// Example 3: GroupBy
	groupedDF, err := df1.GroupBy(
		[]string{"State"},
		[]string{"Purchase"},
		[]string{"avg", "max"},
	)
	if err != nil {
		fmt.Println("Error in GroupBy:", err)
	} else {
		fmt.Println("\nGrouped DataFrame (by State, average and max Purchase):")
		groupedDF.Head(10).Print()
	}

	// Example 4: Unique values in a column
	uniqueStates := df1.Col("State").Unique()
	fmt.Println("\nUnique States:")
	fmt.Println(uniqueStates)

	// Example 5: Sorting
	sortedDF, err := df1.Sort([]string{"Salary"}, []bool{true})
	if err != nil {
		fmt.Println("Error sorting:", err)
	} else {
		fmt.Println("\nSorted DataFrame (by Salary ascending)")
		sortedDF.Head(10).Print()
	}

	// Example 6: Type conversion
	err = df1.AsType("Purchase", "int")
	if err != nil {
		fmt.Println("Error converting Purchase to int:", err)
	} else {
		fmt.Println("\nPurchase column converted to int:")
		fmt.Println(df1.Col("Purchase"))
	}

	// Example 7: Describe
	describeDF := df1.Describe()
	fmt.Println("\nDataFrame Description:")
	describeDF.Head(10).Print()

	// Example 8: Head and Tail
	fmt.Println("\nFirst 5 rows:")
	df1.Head(5).Head(10).Print()
	fmt.Println("\nLast 5 rows:")
	df1.Tail(5).Head(10).Print()

	// Example 9: Drop NA and Fill NA
	cleanDF := df1.DropNA()
	fmt.Println("\nDataFrame with NA values dropped:")
	cleanDF.Head(10).Head(10).Print()

	filledDF := df1.FillNA("0")
	fmt.Println("\nDataFrame with NA values filled with '0':")
	filledDF.Head(10).Head(10).Print()

	// Example 10: Apply custom function
	upperCaseDF := df1.ApplyFunc("Name", strings.ToUpper)
	fmt.Println("\nDataFrame with Names in uppercase:")
	upperCaseDF.Head(10).Print()

	// Example 11: Filtering with FilterCondition slice (for backwards compatibility)
	filterConditions := []FilterCondition{
		{Column: "Name", Operator: "==", Value: "Dominic"},
		{Column: "Purchase", Operator: "<", Value: 500000},
	}
	conditionalFilteredDF := df1.Filter(filterConditions)
	fmt.Println("\nFiltered DataFrame using FilterCondition slice:")
	conditionalFilteredDF.Head(10).Head(10).Print()

	df1.AddColumn("PurchaseCategory", func(row []string) string {
		purchase, _ := strconv.ParseFloat(row[df1.getColumnIndex("Purchase")], 64)
		if purchase > 500000 {
			return "High"
		} else if purchase > 250000 {
			return "Medium"
		} else {
			return "Low"
		}
	})
	fmt.Println("Dataframe with new column")
	df1.Head(10).Head(10).Print()

	// Example 12: Let's do this again, but use logic from another column....

	df1.AddColumn("PurchaseCategoryComplex", func(row []string) string {
		purchase, _ := strconv.ParseFloat(row[df1.getColumnIndex("Purchase")], 64)
		name := row[df1.getColumnIndex("Name")]

		if (purchase <= 1000000) && (name == "Dominic") {
			return "High Dom Purchase"
		} else if name == "Dominic" {
			return "Regular Dom Purchase"

		} else {
			return "Regular Purchase"
		}
	})

	fmt.Println("Let's see if this bad boy works...")
	df1.Head(10).Head(10).Print()

	// Example 13: Drop multiple columns
	dfDropped := df1.Drop("Purchase", "Salary")
	fmt.Println("\nDataFrame after dropping 'Purchase' and 'Salary' columns:")
	dfDropped.Head(10).Head(10).Print()

	// Let's also show it being done with an array....

	dfDropped = df1.Drop([]string{"Purchase", "Salary"})
	fmt.Println("Dropping with string array...")
	dfDropped.Head(10).Head(10).Print()

	// Example 14: Adding Column based on string condition (Pretty Sweet, Huh)

	stringCondition = "(Name == Dominic AND Purchase >= 500000)"
	stringWhereSeries := df1.Where(stringCondition, "Matches String Condition", "Doesn't Match")
	err = df1.AddColumn("String Where Result", stringWhereSeries.data)
	if err != nil {
		fmt.Println("Error adding 'String Where Result' column:", err)
	} else {
		fmt.Println("Added 'String Where Result' column:")
		df1.Head(10).Head(10).Print()
	}

	// Example 15: Complex filtering with string condition
	complexFilteredDF := df1.Filter(stringCondition)
	fmt.Println("\nFiltered DataFrame using complex string condition:")
	complexFilteredDF.Head(10).Head(10).Print()

}
