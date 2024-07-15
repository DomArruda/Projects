package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"

	"log"

	"github.com/xitongsys/parquet-go-source/local"
	"github.com/xitongsys/parquet-go/reader"
)

type DataValue struct {
	Type  string
	Value interface{}
}

type Series struct {
	data  []DataValue
	dtype string
}

type DataFrame struct {
	Columns   []string
	Data      [][]DataValue
	ColumnMap map[string]*Series
}

type FilterCondition struct {
	Column   string
	Operator string
	Value    interface{}
}

func ReadJSONToDataFrame(filename string) (*DataFrame, error) {
	// Read the JSON file
	data, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("error reading JSON file: %v", err)
	}

	// Unmarshal JSON data into a slice of maps
	var jsonData []map[string]interface{}
	err = json.Unmarshal(data, &jsonData)
	if err != nil {
		return nil, fmt.Errorf("error unmarshaling JSON: %v", err)
	}

	// If the JSON is empty, return an empty DataFrame
	if len(jsonData) == 0 {
		return &DataFrame{
			Columns:   []string{},
			Data:      [][]DataValue{},
			ColumnMap: make(map[string]*Series),
		}, nil
	}

	// Extract column names from the first object
	var columns []string
	for key := range jsonData[0] {
		columns = append(columns, key)
	}

	// Create DataFrame data
	dfData := make([][]DataValue, len(jsonData))
	for i, row := range jsonData {
		dfData[i] = make([]DataValue, len(columns))
		for j, col := range columns {
			dfData[i][j] = jsonValueToDataValue(row[col])
		}
	}

	// Create the DataFrame
	df := &DataFrame{
		Columns:   columns,
		Data:      dfData,
		ColumnMap: make(map[string]*Series),
	}

	// Create Series for each column
	for _, col := range columns {
		colData := make([]DataValue, len(jsonData))
		for i, row := range dfData {
			colData[i] = row[df.getColumnIndex(col)]
		}
		df.ColumnMap[col] = NewSeries(colData)
	}

	return df, nil
}

func jsonValueToDataValue(value interface{}) DataValue {
	switch v := value.(type) {
	case nil:
		return DataValue{Type: "null", Value: nil}
	case float64:
		if float64(int(v)) == v {
			return DataValue{Type: "int", Value: int(v)}
		}
		return DataValue{Type: "float", Value: v}
	case string:
		return DataValue{Type: "string", Value: v}
	case bool:
		return DataValue{Type: "bool", Value: v}
	default:
		return DataValue{Type: "string", Value: fmt.Sprintf("%v", v)}
	}
}

func NewDataValue(value string) DataValue {
	if value == "" || value == "NA" || value == "NaN" {
		return DataValue{Type: "null", Value: nil}
	}

	if intVal, err := strconv.Atoi(value); err == nil {
		return DataValue{Type: "int", Value: intVal}
	}

	if floatVal, err := strconv.ParseFloat(value, 64); err == nil {
		return DataValue{Type: "float", Value: floatVal}
	}

	if boolVal, err := strconv.ParseBool(value); err == nil {
		return DataValue{Type: "bool", Value: boolVal}
	}

	return DataValue{Type: "string", Value: value}
}

func NewSeries(data []DataValue) *Series {
	return &Series{
		data:  data,
		dtype: inferDataType(data),
	}
}

func inferDataType(data []DataValue) string {
	if len(data) == 0 {
		return "string"
	}

	isInt := true
	isFloat := true
	isBool := true

	for _, v := range data {
		if v.Type == "null" {
			continue
		}
		switch v.Type {
		case "int":
			isFloat = false
			isBool = false
		case "float":
			isInt = false
			isBool = false
		case "bool":
			isInt = false
			isFloat = false
		default:
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

func (s *Series) toArray() []DataValue {
	return s.data
}

func (s *Series) ToTypedArray() (interface{}, error) {
	switch s.dtype {
	case "int":
		intArray := make([]int, len(s.data))
		for i, v := range s.data {
			if v.Type == "null" {
				continue
			}
			intArray[i] = v.Value.(int)
		}
		return intArray, nil
	case "float":
		floatArray := make([]float64, len(s.data))
		for i, v := range s.data {
			if v.Type == "null" {
				continue
			}
			floatArray[i] = v.Value.(float64)
		}
		return floatArray, nil
	case "bool":
		boolArray := make([]bool, len(s.data))
		for i, v := range s.data {
			if v.Type == "null" {
				continue
			}
			boolArray[i] = v.Value.(bool)
		}
		return boolArray, nil
	default:
		return s.data, nil
	}
}

func (s *Series) Unique() *Series {
	uniqueMap := make(map[string]bool)
	var uniqueData []DataValue
	for _, item := range s.data {
		key := fmt.Sprintf("%v", item.Value)
		if _, exists := uniqueMap[key]; !exists {
			uniqueMap[key] = true
			uniqueData = append(uniqueData, item)
		}
	}
	return NewSeries(uniqueData)
}

func (s *Series) Sum() (float64, error) {
	var sum float64

	for _, v := range s.data {
		if v.Type == "null" {
			continue
		}

		switch v.Type {
		case "int":
			if intVal, ok := v.Value.(int); ok {
				sum += float64(intVal)
			} else {
				return 0, fmt.Errorf("expected int, got %T", v.Value)
			}
		case "float":
			if floatVal, ok := v.Value.(float64); ok {
				sum += floatVal
			} else {
				return 0, fmt.Errorf("expected float64, got %T", v.Value)
			}
		default:
			return 0, fmt.Errorf("sum operation not supported for type: %s", v.Type)
		}
	}

	return sum, nil
}

func (s *Series) Average() (float64, error) {
	var sum float64

	for _, v := range s.data {
		if v.Type == "null" {
			continue
		}

		switch v.Type {
		case "int":
			if intVal, ok := v.Value.(int); ok {
				sum += float64(intVal)
			} else {
				return 0, fmt.Errorf("expected int, got %T", v.Value)
			}
		case "float":
			if floatVal, ok := v.Value.(float64); ok {
				sum += floatVal
			} else {
				return 0, fmt.Errorf("expected float64, got %T", v.Value)
			}
		default:
			return 0, fmt.Errorf("sum operation not supported for type: %s", v.Type)
		}
	}

	dataLength := float64(len((s.data)))
	if dataLength > 0 {
		return sum / dataLength, nil
	} else {
		return 0.0, nil
	}
}

func (s *Series) asType(typeName string) (*Series, error) {
	newData := make([]DataValue, len(s.data))

	for i, value := range s.data {
		if value.Type == "null" {
			newData[i] = value
			continue
		}

		switch typeName {
		case "int":
			intVal, err := strconv.Atoi(fmt.Sprintf("%v", value.Value))
			if err != nil {
				return nil, fmt.Errorf("cannot convert value '%v' to int", value.Value)
			}
			newData[i] = DataValue{Type: "int", Value: intVal}
		case "float":
			floatVal, err := strconv.ParseFloat(fmt.Sprintf("%v", value.Value), 64)
			if err != nil {
				return nil, fmt.Errorf("cannot convert value '%v' to float", value.Value)
			}
			newData[i] = DataValue{Type: "float", Value: floatVal}
		case "string":
			newData[i] = DataValue{Type: "string", Value: fmt.Sprintf("%v", value.Value)}
		case "bool":
			boolVal, err := strconv.ParseBool(fmt.Sprintf("%v", value.Value))
			if err != nil {
				return nil, fmt.Errorf("cannot convert value '%v' to bool", value.Value)
			}
			newData[i] = DataValue{Type: "bool", Value: boolVal}
		default:
			return nil, fmt.Errorf("unsupported type: %s", typeName)
		}
	}

	return &Series{data: newData, dtype: typeName}, nil
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
		ColumnMap: make(map[string]*Series),
	}

	df.Data = make([][]DataValue, len(data)-1)
	for i, row := range data[1:] {
		df.Data[i] = make([]DataValue, len(row))
		for j, cell := range row {
			df.Data[i][j] = NewDataValue(cell)
		}
	}

	for i, colName := range df.Columns {
		column := make([]DataValue, len(df.Data))
		for j, row := range df.Data {
			column[j] = row[i]
		}
		df.ColumnMap[colName] = NewSeries(column)
	}

	return df
}

func evaluateCondition(df *DataFrame, row []DataValue, condition FilterCondition) bool {
	colIndex := df.getColumnIndex(condition.Column)
	if colIndex == -1 {
		return false
	}

	value := row[colIndex]
	if value.Type == "null" {
		return false
	}

	switch condition.Operator {
	case "==":
		return fmt.Sprintf("%v", value.Value) == fmt.Sprintf("%v", condition.Value)
	case "!=":
		return fmt.Sprintf("%v", value.Value) != fmt.Sprintf("%v", condition.Value)
	case ">", "<", ">=", "<=":
		v1, ok1 := toFloat64(value)
		v2, ok2 := toFloat64(DataValue{Type: "string", Value: fmt.Sprintf("%v", condition.Value)})
		if !ok1 || !ok2 {
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
	var newData [][]DataValue

	for _, row := range df.Data {
		key := make([]string, len(subset))
		for i, col := range subset {
			key[i] = fmt.Sprintf("%v", row[df.getColumnIndex(col)].Value)
		}
		keyStr := strings.Join(key, "|")
		if !seen[keyStr] {
			seen[keyStr] = true
			newData = append(newData, row)
		}
	}

	newDF := &DataFrame{
		Columns:   df.Columns,
		Data:      newData,
		ColumnMap: make(map[string]*Series),
	}

	for _, colName := range df.Columns {
		newDF.ColumnMap[colName] = NewSeries(df.Col(colName).data)
	}

	return newDF
}

func (df *DataFrame) AddColumn(name string, data interface{}) error {
	var newColumn []DataValue

	switch v := data.(type) {
	case func(row []DataValue) DataValue:
		newColumn = make([]DataValue, len(df.Data))
		for i, row := range df.Data {
			newColumn[i] = v(row)
		}
	case []DataValue:
		if len(v) != len(df.Data) {
			return fmt.Errorf("array length (%d) does not match DataFrame length (%d)", len(v), len(df.Data))
		}
		newColumn = v
	case []string:
		if len(v) != len(df.Data) {
			return fmt.Errorf("array length (%d) does not match DataFrame length (%d)", len(v), len(df.Data))
		}
		newColumn = make([]DataValue, len(v))
		for i, val := range v {
			newColumn[i] = NewDataValue(val)
		}
	case []int:
		if len(v) != len(df.Data) {
			return fmt.Errorf("array length (%d) does not match DataFrame length (%d)", len(v), len(df.Data))
		}
		newColumn = make([]DataValue, len(v))
		for i, val := range v {
			newColumn[i] = DataValue{Type: "int", Value: val}
		}
	case []float64:
		if len(v) != len(df.Data) {
			return fmt.Errorf("array length (%d) does not match DataFrame length (%d)", len(v), len(df.Data))
		}
		newColumn = make([]DataValue, len(v))
		for i, val := range v {
			newColumn[i] = DataValue{Type: "float", Value: val}
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

func (df *DataFrame) UpdateColumn(name string, updateFunc func(value DataValue) DataValue) error {
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
	result := make([]DataValue, len(df.Data))

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

func evaluateStringCondition(df *DataFrame, row []DataValue, condition string) bool {
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

func evaluateSimpleCondition(df *DataFrame, row []DataValue, condition string) bool {
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
		return fmt.Sprintf("%v", cellValue.Value) == value
	case "!=":
		return fmt.Sprintf("%v", cellValue.Value) != value
	case ">", "<", ">=", "<=":
		cellFloat, ok1 := toFloat64(cellValue)
		valueFloat, err := strconv.ParseFloat(value, 64)
		if !ok1 || err != nil {
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

func evaluateFilterConditions(df *DataFrame, row []DataValue, conditions []FilterCondition) bool {

	for _, condition := range conditions {
		if !evaluateCondition(df, row, condition) {
			return false
		}
	}
	return true
}

func (df *DataFrame) Filter(condition interface{}) *DataFrame {
	var filteredData [][]DataValue

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

	newDF := &DataFrame{
		Columns:   df.Columns,
		Data:      filteredData,
		ColumnMap: make(map[string]*Series),
	}

	for _, colName := range df.Columns {
		newCol := make([]DataValue, len(filteredData))
		for i, row := range filteredData {
			newCol[i] = row[df.getColumnIndex(colName)]
		}
		newDF.ColumnMap[colName] = NewSeries(newCol)
	}

	return newDF
}

func applyValue(value interface{}, row []DataValue) DataValue {
	switch v := value.(type) {
	case string:
		return DataValue{Type: "string", Value: v}
	case func([]DataValue) DataValue:
		return v(row)
	case int:
		return DataValue{Type: "int", Value: v}
	case float64:
		return DataValue{Type: "float", Value: v}
	case bool:
		return DataValue{Type: "bool", Value: v}
	default:
		return DataValue{Type: "string", Value: fmt.Sprintf("%v", v)}
	}
}

// DataFrame Methods

func (df *DataFrame) Rename(oldName, newName string) error {
	index := df.getColumnIndex(oldName)
	if index == -1 {
		return fmt.Errorf("column '%s' not found", oldName)
	}
	df.Columns[index] = newName
	df.ColumnMap[newName] = df.ColumnMap[oldName]
	delete(df.ColumnMap, oldName)
	return nil
}

func (df *DataFrame) Concat(other *DataFrame) *DataFrame {
	if df == nil {
		return other
	}
	if other == nil {
		return df
	}

	newColumns := make([]string, len(df.Columns))
	copy(newColumns, df.Columns)

	for _, col := range other.Columns {
		if df.getColumnIndex(col) == -1 {
			newColumns = append(newColumns, col)
		}
	}

	newData := make([][]DataValue, len(df.Data)+len(other.Data))
	copy(newData, df.Data)

	for i, row := range other.Data {
		newRow := make([]DataValue, len(newColumns))
		for j, col := range newColumns {
			if index := other.getColumnIndex(col); index != -1 && index < len(row) {
				newRow[j] = row[index]
			} else {
				newRow[j] = DataValue{Type: "null", Value: nil}
			}
		}
		newData[len(df.Data)+i] = newRow
	}

	newDF := &DataFrame{
		Columns:   newColumns,
		Data:      newData,
		ColumnMap: make(map[string]*Series),
	}

	for _, col := range newColumns {
		newCol := make([]DataValue, len(newData))
		for i, row := range newData {
			colIndex := newDF.getColumnIndex(col)
			if colIndex != -1 && colIndex < len(row) {
				newCol[i] = row[colIndex]
			} else {
				newCol[i] = DataValue{Type: "null", Value: nil}
			}
		}
		newDF.ColumnMap[col] = NewSeries(newCol)
	}

	return newDF
}

func (df *DataFrame) Cumsum() *DataFrame {
	newDF := &DataFrame{
		Columns:   df.Columns,
		Data:      make([][]DataValue, len(df.Data)),
		ColumnMap: make(map[string]*Series),
	}

	for i, col := range df.Columns {
		series := df.Col(col)
		if series.dtype == "float" || series.dtype == "int" {
			newSeries := series.Cumsum()
			newDF.ColumnMap[col] = newSeries

			for j := range newDF.Data {
				if newDF.Data[j] == nil {
					newDF.Data[j] = make([]DataValue, len(df.Columns))
				}
				newDF.Data[j][i] = newSeries.data[j]
			}
		}
	}

	return newDF
}

// Series Methods

func (s *Series) Cumsum() *Series {
	newData := make([]DataValue, len(s.data))
	var sum float64

	for i, v := range s.data {
		if v.Type != "null" {
			f, ok := toFloat64(v)
			if ok {
				sum += f
				newData[i] = DataValue{Type: "float", Value: sum}
			} else {
				newData[i] = DataValue{Type: "null", Value: nil}
			}
		} else {
			newData[i] = DataValue{Type: "null", Value: nil}
		}
	}

	return &Series{data: newData, dtype: "float"}
}

func (s *Series) Mode() ([]interface{}, error) {
	if s == nil {
		return nil, fmt.Errorf("series is nil")
	}

	if len(s.data) == 0 {
		return nil, fmt.Errorf("series is empty")
	}

	counts := make(map[interface{}]int)
	for _, v := range s.data {
		if v.Type != "null" {
			counts[v.Value]++
		}
	}

	if len(counts) == 0 {
		return nil, fmt.Errorf("no non-null values in series")
	}

	maxCount := 0
	for _, count := range counts {
		if count > maxCount {
			maxCount = count
		}
	}

	var modes []interface{}
	for value, count := range counts {
		if count == maxCount {
			modes = append(modes, value)
		}
	}

	return modes, nil
}
func (s *Series) Quantile(q float64) (float64, error) {
	if s == nil {
		return 0, fmt.Errorf("series is nil")
	}

	if q < 0 || q > 1 {
		return 0, fmt.Errorf("quantile must be between 0 and 1")
	}

	var floatArray []float64
	for _, val := range s.data {
		if val.Type != "null" {
			f, ok := toFloat64(val)
			if ok {
				floatArray = append(floatArray, f)
			}
		}
	}

	if len(floatArray) == 0 {
		return 0, fmt.Errorf("no numeric data in series")
	}

	sort.Float64s(floatArray)

	index := q * float64(len(floatArray)-1)
	lower := math.Floor(index)
	upper := math.Ceil(index)

	if lower == upper {
		return floatArray[int(lower)], nil
	}

	lowerValue := floatArray[int(lower)]
	upperValue := floatArray[int(upper)]

	interpolation := index - lower
	result := lowerValue + interpolation*(upperValue-lowerValue)

	return result, nil
}

func (s *Series) Median() (float64, error) {
	if s == nil {
		return 0, fmt.Errorf("series is nil")
	}
	return s.Quantile(0.5)
}

func (s *Series) Nsmallest(n int) (*Series, error) {
	values := make([]DataValue, len(s.data))
	copy(values, s.data)

	sort.Slice(values, func(i, j int) bool {
		if values[i].Type == "null" {
			return false
		}
		if values[j].Type == "null" {
			return true
		}
		v1, _ := toFloat64(values[i])
		v2, _ := toFloat64(values[j])
		return v1 < v2
	})

	if n > len(values) {
		n = len(values)
	}

	return &Series{data: values[:n], dtype: s.dtype}, nil
}

func (s *Series) Nlargest(n int) (*Series, error) {
	if s == nil {
		return nil, fmt.Errorf("series is nil")
	}

	values := make([]DataValue, len(s.data))
	copy(values, s.data)

	sort.Slice(values, func(i, j int) bool {
		if values[i].Type == "null" {
			return false
		}
		if values[j].Type == "null" {
			return true
		}
		v1, ok1 := toFloat64(values[i])
		v2, ok2 := toFloat64(values[j])
		if !ok1 || !ok2 {
			return false
		}
		return v1 > v2
	})

	if n > len(values) {
		n = len(values)
	}

	return &Series{data: values[:n], dtype: s.dtype}, nil
}

func (df *DataFrame) GroupBy(categoricalCols []string, numericalCols []string, aggregations []string) (*DataFrame, error) {
	groups := make(map[string][][]DataValue)

	for _, row := range df.Data {
		key := make([]string, len(categoricalCols))
		for i, col := range categoricalCols {
			colIndex := df.getColumnIndex(col)
			if colIndex == -1 {
				return nil, fmt.Errorf("categorical column not found: %s", col)
			}
			key[i] = fmt.Sprintf("%v", row[colIndex].Value)
		}
		keyStr := strings.Join(key, "|")
		groups[keyStr] = append(groups[keyStr], row)
	}

	var newData [][]DataValue
	newColumns := append(categoricalCols, make([]string, 0, len(numericalCols)*len(aggregations))...)
	for _, numCol := range numericalCols {
		for _, agg := range aggregations {
			newColumns = append(newColumns, fmt.Sprintf("%s_%s", numCol, agg))
		}
	}

	for _, rows := range groups {
		newRow := make([]DataValue, len(newColumns))
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
			values := make([]float64, 0, len(rows))
			for _, row := range rows {
				if row[colIndex].Type != "null" {
					val, ok := toFloat64(row[colIndex])
					if !ok {
						return nil, fmt.Errorf("invalid numeric value in column %s: %v", numCol, row[colIndex].Value)
					}
					values = append(values, val)
				}
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
				newRow[i] = DataValue{Type: "float", Value: result}
				i++
			}
		}

		newData = append(newData, newRow)
	}

	newDF := &DataFrame{
		Columns:   newColumns,
		Data:      newData,
		ColumnMap: make(map[string]*Series),
	}

	for _, colName := range newColumns {
		newCol := make([]DataValue, len(newData))
		for i, row := range newData {
			newCol[i] = row[newDF.getColumnIndex(colName)]
		}
		newDF.ColumnMap[colName] = NewSeries(newCol)
	}

	return newDF, nil
}

func getValueOrNull(dv DataValue) interface{} {
	if dv.Type == "null" {
		return nil
	}
	return dv.Value
}

func (df *DataFrame) Join(other *DataFrame, joinType string, leftOn, rightOn string) (*DataFrame, error) {
	leftIndex := df.getColumnIndex(leftOn)
	rightIndex := other.getColumnIndex(rightOn)
	if leftIndex == -1 || rightIndex == -1 {
		return nil, fmt.Errorf("join columns not found")
	}

	leftMap := make(map[string][]int)
	for i, row := range df.Data {
		key := fmt.Sprintf("%v", getValueOrNull(row[leftIndex]))
		leftMap[key] = append(leftMap[key], i)
	}

	var newColumns []string
	newColumns = append(newColumns, df.Columns...)
	for _, col := range other.Columns {
		if col != rightOn {
			newColumns = append(newColumns, col)
		}
	}

	var newData [][]DataValue
	switch joinType {
	case "inner":
		for _, rightRow := range other.Data {
			rightKey := fmt.Sprintf("%v", getValueOrNull(rightRow[rightIndex]))
			if leftIndices, ok := leftMap[rightKey]; ok {
				for _, leftIndex := range leftIndices {
					newRow := append([]DataValue{}, df.Data[leftIndex]...)
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
		for _, leftRow := range df.Data {
			leftKey := fmt.Sprintf("%v", getValueOrNull(leftRow[leftIndex]))
			if _, ok := leftMap[leftKey]; ok {
				matched := false
				for _, rightRow := range other.Data {
					rightKey := fmt.Sprintf("%v", getValueOrNull(rightRow[rightIndex]))
					if leftKey == rightKey {
						newRow := append([]DataValue{}, leftRow...)
						for j, val := range rightRow {
							if j != rightIndex {
								newRow = append(newRow, val)
							}
						}
						newData = append(newData, newRow)
						matched = true
					}
				}
				if !matched {
					newRow := append([]DataValue{}, leftRow...)
					for range other.Columns {
						if len(newRow) < len(newColumns) {
							newRow = append(newRow, DataValue{Type: "null", Value: nil})
						}
					}
					newData = append(newData, newRow)
				}
			} else {
				newRow := append([]DataValue{}, leftRow...)
				for range other.Columns {
					if len(newRow) < len(newColumns) {
						newRow = append(newRow, DataValue{Type: "null", Value: nil})
					}
				}
				newData = append(newData, newRow)
			}
		}
	case "outer":
		// Create a map to keep track of matched rows from the right DataFrame
		rightMatched := make(map[string]bool)

		// First, perform a left join
		for _, leftRow := range df.Data {
			leftKey := fmt.Sprintf("%v", getValueOrNull(leftRow[leftIndex]))
			matched := false
			for _, rightRow := range other.Data {
				rightKey := fmt.Sprintf("%v", getValueOrNull(rightRow[rightIndex]))
				if leftKey == rightKey {
					newRow := append([]DataValue{}, leftRow...)
					for j, val := range rightRow {
						if j != rightIndex {
							newRow = append(newRow, val)
						}
					}
					newData = append(newData, newRow)
					matched = true
					rightMatched[rightKey] = true
				}
			}
			if !matched {
				newRow := append([]DataValue{}, leftRow...)
				for range other.Columns {
					if len(newRow) < len(newColumns) {
						newRow = append(newRow, DataValue{Type: "null", Value: nil})
					}
				}
				newData = append(newData, newRow)
			}
		}

		// Then, add any unmatched rows from the right DataFrame
		for _, rightRow := range other.Data {
			rightKey := fmt.Sprintf("%v", getValueOrNull(rightRow[rightIndex]))
			if !rightMatched[rightKey] {
				newRow := make([]DataValue, len(newColumns))
				for i := range df.Columns {
					newRow[i] = DataValue{Type: "null", Value: nil}
				}
				newRow[leftIndex] = rightRow[rightIndex]
				rightColIndex := len(df.Columns)
				for j, val := range rightRow {
					if j != rightIndex {
						newRow[rightColIndex] = val
						rightColIndex++
					}
				}
				newData = append(newData, newRow)
			}
		}
	default:
		return nil, fmt.Errorf("unsupported join type: %s", joinType)
	}

	newDF := &DataFrame{
		Columns:   newColumns,
		Data:      newData,
		ColumnMap: make(map[string]*Series),
	}

	for _, colName := range newColumns {
		newCol := make([]DataValue, len(newData))
		for i, row := range newData {
			colIndex := newDF.getColumnIndex(colName)
			if colIndex < len(row) {
				newCol[i] = row[colIndex]
			} else {
				newCol[i] = DataValue{Type: "null", Value: nil}
			}
		}
		newDF.ColumnMap[colName] = NewSeries(newCol)
	}

	return newDF, nil
}

func (df *DataFrame) Sort(columns []string, ascending []bool) (*DataFrame, error) {
	if len(columns) != len(ascending) {
		return nil, fmt.Errorf("length of columns and ascending must match")
	}

	sortedDF := &DataFrame{
		Columns:   df.Columns,
		Data:      make([][]DataValue, len(df.Data)),
		ColumnMap: make(map[string]*Series),
	}
	copy(sortedDF.Data, df.Data)

	sort.SliceStable(sortedDF.Data, func(i, j int) bool {
		for k, col := range columns {
			colIndex := sortedDF.getColumnIndex(col)
			if colIndex == -1 {
				return false
			}

			cmp := compareDataValues(sortedDF.Data[i][colIndex], sortedDF.Data[j][colIndex])
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

	for _, colName := range df.Columns {
		newCol := make([]DataValue, len(sortedDF.Data))
		for i, row := range sortedDF.Data {
			newCol[i] = row[sortedDF.getColumnIndex(colName)]
		}
		sortedDF.ColumnMap[colName] = NewSeries(newCol)
	}

	return sortedDF, nil
}

func compareDataValues(a, b DataValue) int {
	if a.Type == "null" && b.Type == "null" {
		return 0
	}
	if a.Type == "null" {
		return -1
	}
	if b.Type == "null" {
		return 1
	}

	switch a.Type {
	case "int":
		return compareInts(a.Value.(int), b.Value.(int))
	case "float":
		return compareFloats(a.Value.(float64), b.Value.(float64))
	case "string":
		return strings.Compare(a.Value.(string), b.Value.(string))
	case "bool":
		return compareBools(a.Value.(bool), b.Value.(bool))
	default:
		return 0
	}
}

func compareInts(a, b int) int {
	if a < b {
		return -1
	}
	if a > b {
		return 1
	}
	return 0
}

func compareFloats(a, b float64) int {
	if a < b {
		return -1
	}
	if a > b {
		return 1
	}
	return 0
}

func compareBools(a, b bool) int {
	if a == b {
		return 0
	}
	if a {
		return 1
	}
	return -1
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
	newDF := &DataFrame{
		Columns:   df.Columns,
		Data:      df.Data[:n],
		ColumnMap: make(map[string]*Series),
	}

	for _, colName := range df.Columns {
		newCol := make([]DataValue, n)
		for i := 0; i < n; i++ {
			newCol[i] = df.Data[i][df.getColumnIndex(colName)]
		}
		newDF.ColumnMap[colName] = NewSeries(newCol)
	}

	return newDF
}

func (s *Series) Print() {
	fmt.Println("Series:")
	for i, v := range s.data {
		if i >= 10 {
			fmt.Println("...")
			break
		}
		if v.Type == "null" {
			fmt.Printf("%d: null\n", i)
		} else {
			fmt.Printf("%d: %v\n", i, v.Value)
		}
	}
	fmt.Printf("Length: %d, Type: %s\n", len(s.data), s.dtype)
}

func (s *Series) Head(n int) *Series {
	if n > len(s.data) {
		n = len(s.data)
	}

	newData := make([]DataValue, n)
	copy(newData, s.data[:n])

	return &Series{
		data:  newData,
		dtype: s.dtype,
	}
}

func (df *DataFrame) Tail(n int) *DataFrame {
	if n > len(df.Data) {
		n = len(df.Data)
	}
	start := len(df.Data) - n
	newDF := &DataFrame{
		Columns:   df.Columns,
		Data:      df.Data[start:],
		ColumnMap: make(map[string]*Series),
	}

	for _, colName := range df.Columns {
		newCol := make([]DataValue, n)
		for i := 0; i < n; i++ {
			newCol[i] = df.Data[start+i][df.getColumnIndex(colName)]
		}
		newDF.ColumnMap[colName] = NewSeries(newCol)
	}

	return newDF
}

func calculateSum(data []float64) float64 {
	sum := 0.0
	for _, v := range data {
		sum += v
	}
	return sum
}

func calculateMean(data []float64) float64 {
	if len(data) == 0 {
		return 0
	}
	return calculateSum(data) / float64(len(data))
}

func calculateStdDev(data []float64, mean float64) float64 {
	if len(data) == 0 {
		return 0
	}
	sumSquaredDiff := 0.0
	for _, v := range data {
		diff := v - mean
		sumSquaredDiff += diff * diff
	}
	variance := sumSquaredDiff / float64(len(data))
	return math.Sqrt(variance)
}

func calculateMin(data []float64) float64 {
	if len(data) == 0 {
		return 0
	}
	min := data[0]
	for _, v := range data[1:] {
		if v < min {
			min = v
		}
	}
	return min
}

func calculateMax(data []float64) float64 {
	if len(data) == 0 {
		return 0
	}
	max := data[0]
	for _, v := range data[1:] {
		if v > max {
			max = v
		}
	}
	return max
}

func calculatePercentile(sortedData []float64, percentile float64) float64 {
	if len(sortedData) == 0 {
		return 0
	}
	index := percentile * float64(len(sortedData)-1)
	lower := math.Floor(index)
	upper := math.Ceil(index)
	if lower == upper {
		return sortedData[int(lower)]
	}
	return sortedData[int(lower)]*(upper-index) + sortedData[int(upper)]*(index-lower)
}

func toFloat64(v DataValue) (float64, bool) {
	switch v.Type {
	case "int":
		return float64(v.Value.(int)), true
	case "float":
		return v.Value.(float64), true
	case "string":
		f, err := strconv.ParseFloat(v.Value.(string), 64)
		return f, err == nil
	default:
		return 0, false
	}
}

func (s *Series) Percentile(percentile float64) (float64, error) {
	if percentile < 0 || percentile > 100 {
		return 0, nil
	}

	var floatArray []float64
	for _, val := range s.data {
		if val.Type == "float" || val.Type == "integer" {
			f, ok := toFloat64(val)
			if ok {
				floatArray = append(floatArray, f)
			}
		}
	}

	if len(floatArray) == 0 {
		return 0, nil
	}

	sort.Float64s(floatArray)

	index := (percentile / 100) * float64(len(floatArray)-1)
	lower := math.Floor(index)
	upper := math.Ceil(index)

	if lower == upper {
		return floatArray[int(lower)], nil
	}

	lowerValue := floatArray[int(lower)]
	upperValue := floatArray[int(upper)]

	interpolation := index - lower
	result := lowerValue + interpolation*(upperValue-lowerValue)

	return result, nil
}

// Update other methods to handle null values properly

func (s *Series) String() string {
	values := make([]string, len(s.data))
	for i, v := range s.data {
		if v.Type == "null" {
			values[i] = "null"
		} else {
			values[i] = fmt.Sprintf("%v", v.Value)
		}
	}
	return "[" + strings.Join(values, ", ") + "]"
}

func (df *DataFrame) Print() {
	fmt.Println("\n")
	colWidths := make([]int, len(df.Columns))
	for i, col := range df.Columns {
		colWidths[i] = len(col)
	}
	for _, row := range df.Data {
		for i, cell := range row {
			cellStr := ""
			if cell.Type == "null" {
				cellStr = "null"
			} else {
				cellStr = fmt.Sprintf("%v", cell.Value)
			}
			if len(cellStr) > colWidths[i] {
				colWidths[i] = len(cellStr)
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
			cellStr := ""
			if cell.Type == "null" {
				cellStr = "null"
			} else {
				cellStr = fmt.Sprintf("%v", cell.Value)
			}
			fmt.Printf("%-*s\t", colWidths[i], cellStr)
		}
		fmt.Println()
	}
}
func (df *DataFrame) Describe() *DataFrame {
	stats := []string{"count", "mean", "std", "min", "25%", "50%", "75%", "max"}

	// First, identify numeric columns
	numericColumns := []string{}
	for _, col := range df.Columns {
		series := df.Col(col)
		if series.dtype == "int" || series.dtype == "float" {
			numericColumns = append(numericColumns, col)
		}
	}

	newData := make([][]DataValue, len(stats))
	for i := range newData {
		newData[i] = make([]DataValue, len(numericColumns)+1)
		newData[i][0] = DataValue{Type: "string", Value: stats[i]}
	}

	for j, col := range numericColumns {
		series := df.Col(col)
		floats := make([]float64, 0, len(series.data))
		count := 0
		for _, v := range series.data {
			if v.Type != "null" {
				f, ok := toFloat64(v)
				if ok {
					floats = append(floats, f)
					count++
				}
			}
		}

		newData[0][j+1] = DataValue{Type: "int", Value: count} // count

		if count == 0 {
			for i := 1; i < len(stats); i++ {
				newData[i][j+1] = DataValue{Type: "null", Value: nil}
			}
			continue
		}

		mean := calculateMean(floats)
		std := calculateStdDev(floats, mean)
		min := calculateMin(floats)
		max := calculateMax(floats)

		sort.Float64s(floats)

		q1 := calculatePercentile(floats, 0.25)
		median := calculatePercentile(floats, 0.5)
		q3 := calculatePercentile(floats, 0.75)

		newData[1][j+1] = DataValue{Type: "float", Value: formatFloat(mean, 6)}
		newData[2][j+1] = DataValue{Type: "float", Value: formatFloat(std, 6)}
		newData[3][j+1] = DataValue{Type: "float", Value: formatFloat(min, 6)}
		newData[4][j+1] = DataValue{Type: "float", Value: formatFloat(q1, 6)}
		newData[5][j+1] = DataValue{Type: "float", Value: formatFloat(median, 6)}
		newData[6][j+1] = DataValue{Type: "float", Value: formatFloat(q3, 6)}
		newData[7][j+1] = DataValue{Type: "float", Value: formatFloat(max, 6)}
	}

	newColumns := append([]string{""}, numericColumns...)
	return &DataFrame{
		Columns:   newColumns,
		Data:      newData,
		ColumnMap: make(map[string]*Series),
	}
}

func formatFloat(f float64, precision int) float64 {
	format := fmt.Sprintf("%%.%df", precision)
	s := fmt.Sprintf(format, f)
	result, _ := strconv.ParseFloat(s, 64)
	return result
}

func (df *DataFrame) CountNonNull() map[string]int {
	counts := make(map[string]int)
	for _, col := range df.Columns {
		count := 0
		for _, v := range df.Col(col).data {
			if v.Type != "null" {
				count++
			}
		}
		counts[col] = count
	}
	return counts
}

func (s *Series) CountNonNull() int64 {
	var count int64
	for _, v := range s.data {
		if v.Type != "null" {
			count++
		}
	}
	return count
}

func (df *DataFrame) FillNA(values map[string]interface{}) *DataFrame {
	newData := make([][]DataValue, len(df.Data))
	for i, row := range df.Data {
		newRow := make([]DataValue, len(row))
		for j, cell := range row {
			if cell.Type == "null" {
				colName := df.Columns[j]
				if fillValue, ok := values[colName]; ok {
					newRow[j] = toDataValue(fillValue)
				} else {
					newRow[j] = cell
				}
			} else {
				newRow[j] = cell
			}
		}
		newData[i] = newRow
	}

	newDF := &DataFrame{
		Columns:   df.Columns,
		Data:      newData,
		ColumnMap: make(map[string]*Series),
	}

	for _, colName := range df.Columns {
		newDF.ColumnMap[colName] = NewSeries(newDF.Col(colName).data)
	}

	return newDF
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

	newData := make([][]DataValue, len(df.Data))
	for i, row := range df.Data {
		newRow := make([]DataValue, len(columnIndices))
		for j, index := range columnIndices {
			newRow[j] = row[index]
		}
		newData[i] = newRow
	}

	newDF := &DataFrame{
		Columns:   newColumns,
		Data:      newData,
		ColumnMap: make(map[string]*Series),
	}

	for _, colName := range newColumns {
		newDF.ColumnMap[colName] = NewSeries(df.Col(colName).data)
	}

	return newDF
}

func (df *DataFrame) Drop(columns ...interface{}) *DataFrame {
	var dropColumns []string

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

	newData := make([][]DataValue, len(df.Data))
	for i, row := range df.Data {
		newRow := make([]DataValue, len(columnIndices))
		for j, index := range columnIndices {
			newRow[j] = row[index]
		}
		newData[i] = newRow
	}

	newDF := &DataFrame{
		Columns:   newColumns,
		Data:      newData,
		ColumnMap: make(map[string]*Series),
	}

	for _, colName := range newColumns {
		newDF.ColumnMap[colName] = NewSeries(df.Col(colName).data)
	}

	return newDF
}

func (df *DataFrame) DropNA() *DataFrame {
	var newData [][]DataValue
	for _, row := range df.Data {
		hasNA := false
		for _, cell := range row {
			if cell.Type == "null" {
				hasNA = true
				break
			}
		}
		if !hasNA {
			newData = append(newData, row)
		}
	}

	newDF := &DataFrame{
		Columns:   df.Columns,
		Data:      newData,
		ColumnMap: make(map[string]*Series),
	}

	for _, colName := range df.Columns {
		newCol := make([]DataValue, len(newData))
		for i, row := range newData {
			newCol[i] = row[newDF.getColumnIndex(colName)]
		}
		newDF.ColumnMap[colName] = NewSeries(newCol)
	}

	return newDF
}
func (s *Series) FillNA(value interface{}) *Series {
	newData := make([]DataValue, len(s.data))
	fillValue := toDataValue(value)

	for i, v := range s.data {
		if v.Type == "null" {
			newData[i] = fillValue
		} else {
			newData[i] = v
		}
	}

	return &Series{
		data:  newData,
		dtype: s.dtype,
	}
}

// Helper function to convert interface{} to DataValue
func toDataValue(value interface{}) DataValue {
	switch v := value.(type) {
	case int:
		return DataValue{Type: "int", Value: v}
	case float64:
		return DataValue{Type: "float", Value: v}
	case string:
		return DataValue{Type: "string", Value: v}
	case bool:
		return DataValue{Type: "bool", Value: v}
	default:
		return DataValue{Type: "string", Value: fmt.Sprintf("%v", v)}
	}
}

func (df *DataFrame) Merge(other *DataFrame, on string, how string) (*DataFrame, error) {
	return df.Join(other, how, on, on)
}
func (df *DataFrame) ApplyFunc(column string, f func(string) string) *DataFrame {
	colIndex := df.getColumnIndex(column)
	if colIndex == -1 {
		return df
	}

	newData := make([][]DataValue, len(df.Data))
	for i, row := range df.Data {
		newRow := make([]DataValue, len(row))
		copy(newRow, row)
		if row[colIndex].Type == "string" {
			newRow[colIndex] = DataValue{
				Type:  "string",
				Value: f(row[colIndex].Value.(string)),
			}
		}
		newData[i] = newRow
	}

	newDF := &DataFrame{
		Columns:   df.Columns,
		Data:      newData,
		ColumnMap: make(map[string]*Series),
	}

	for _, colName := range df.Columns {
		newCol := make([]DataValue, len(newData))
		for i, row := range newData {
			newCol[i] = row[newDF.getColumnIndex(colName)]
		}
		newDF.ColumnMap[colName] = NewSeries(newCol)
	}

	return newDF
}

func ReadCSVStringToDataFrame(csvString string) (*DataFrame, error) {
	reader := csv.NewReader(strings.NewReader(csvString))
	reader.TrimLeadingSpace = true // This will trim leading spaces from fields
	data, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	return NewDataFrame(data), nil
}
func ReadParquetToDataFrame(filename string) (*DataFrame, error) {
	fr, err := local.NewLocalFileReader(filename)
	if err != nil {
		return nil, fmt.Errorf("can't open file: %s", err)
	}
	defer fr.Close()

	pr, err := reader.NewParquetReader(fr, nil, 4)
	if err != nil {
		return nil, fmt.Errorf("can't create parquet reader: %s", err)
	}
	defer pr.ReadStop()

	num := int(pr.GetNumRows())
	res := make([]map[string]interface{}, num)

	for i := 0; i < num; i++ {
		row := make(map[string]interface{})
		if err = pr.Read(&row); err != nil {
			return nil, fmt.Errorf("can't read row: %s", err)
		}
		res[i] = row
	}

	// Convert the result to DataFrame
	if len(res) == 0 {
		return &DataFrame{
			Columns:   []string{},
			Data:      [][]DataValue{},
			ColumnMap: make(map[string]*Series),
		}, nil
	}

	var columns []string
	for key := range res[0] {
		columns = append(columns, key)
	}

	dfData := make([][]DataValue, len(res))
	for i, row := range res {
		dfData[i] = make([]DataValue, len(columns))
		for j, col := range columns {
			dfData[i][j] = jsonValueToDataValue(row[col])
		}
	}

	df := &DataFrame{
		Columns:   columns,
		Data:      dfData,
		ColumnMap: make(map[string]*Series),
	}

	for _, col := range columns {
		colData := make([]DataValue, len(res))
		for i, row := range dfData {
			colData[i] = row[df.getColumnIndex(col)]
		}
		df.ColumnMap[col] = NewSeries(colData)
	}

	return df, nil
}

func main() {
	// Example 1: Reading CSV and basic operations
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
	uniqueStates.Print()

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
		df1.Col("Purchase").Print()
	}

	// Example 7: Describe
	describeDF := df1.Describe()
	fmt.Println("\nDataFrame Description:")
	describeDF.Print()

	// Example 8: Head and Tail
	fmt.Println("\nFirst 5 rows:")
	df1.Head(5).Print()
	fmt.Println("\nLast 5 rows:")
	df1.Tail(5).Print()

	// Example 9: Drop NA
	cleanDF := df1.DropNA()
	fmt.Println("\nDataFrame with NA values dropped:")
	cleanDF.Head(10).Print()

	// Example 10: Apply custom function
	upperCaseDF := df1.ApplyFunc("Name", strings.ToUpper)
	fmt.Println("\nDataFrame with Names in uppercase:")
	upperCaseDF.Head(10).Print()

	// Example 11: Filtering with FilterCondition slice
	filterConditions := []FilterCondition{
		{Column: "Name", Operator: "==", Value: "Dominic"},
		{Column: "Purchase", Operator: "<", Value: 500000},
	}
	conditionalFilteredDF := df1.Filter(filterConditions)
	fmt.Println("\nFiltered DataFrame using FilterCondition slice:")
	conditionalFilteredDF.Head(10).Print()

	// Example 12: Adding new columns with complex logic
	df1.AddColumn("PurchaseCategory", func(row []DataValue) DataValue {
		purchase, _ := strconv.ParseFloat(fmt.Sprintf("%v", row[df1.getColumnIndex("Purchase")].Value), 64)
		if purchase > 500000 {
			return DataValue{Type: "string", Value: "High"}
		} else if purchase > 250000 {
			return DataValue{Type: "string", Value: "Medium"}
		} else {
			return DataValue{Type: "string", Value: "Low"}
		}
	})
	fmt.Println("\nDataframe with new PurchaseCategory column:")
	df1.Head(10).Print()

	// Example 13: Drop multiple columns
	dfDropped := df1.Drop("Purchase", "Salary")
	fmt.Println("\nDataFrame after dropping 'Purchase' and 'Salary' columns:")
	dfDropped.Head(10).Print()

	// Example 14: Adding Column based on string condition
	stringCondition = "(Name == Dominic AND Purchase >= 500000)"
	stringWhereSeries := df1.Where(stringCondition, "Matches String Condition", "Doesn't Match")
	err = df1.AddColumn("String Where Result", stringWhereSeries.data)
	if err != nil {
		fmt.Println("Error adding 'String Where Result' column:", err)
	} else {
		fmt.Println("\nAdded 'String Where Result' column:")
		df1.Head(10).Print()
	}

	// Example 15: Complex filtering with string condition
	complexFilteredDF := df1.Filter(stringCondition)
	fmt.Println("\nFiltered DataFrame using complex string condition:")
	complexFilteredDF.Head(10).Print()

	// Example 16: Creating a DataFrame From CSV String and Dropping Nulls
	csvString := `Name,Purchase
    Dominic,125
    ,200
    Alex,300
    Phil,`

	dfFromCSV, err := ReadCSVStringToDataFrame(csvString)
	if err != nil {
		fmt.Println("Error reading CSV string:", err)
	} else {
		fmt.Println("\nDataFrame created from CSV string:")
		dfFromCSV.Print()
	}

	// Example 17: Reading JSON into dataframe
	jsonFilename := "data.json"
	jsonDF, err := ReadJSONToDataFrame(jsonFilename)
	if err != nil {
		fmt.Println("Error reading JSON to DataFrame:", err)
	} else {
		fmt.Println("\nDataFrame from JSON:")
		jsonDF.Print()
	}

	// Example 18: Joining DataFrames
	leftDFString := `ProductID,Transaction
    1,20
    2,30
    3,40
    1,50
    4,30`

	rightDFString := `ProductID,Name
    1,Book
    2,Espresso
    3,Sandwich
    5,Capybara`

	leftDF, err := ReadCSVStringToDataFrame(leftDFString)
	if err != nil {
		fmt.Println("Error creating left DataFrame:", err)
		return
	}

	rightDF, err := ReadCSVStringToDataFrame(rightDFString)
	if err != nil {
		fmt.Println("Error creating right DataFrame:", err)
		return
	}

	leftJoinedDF, err := leftDF.Join(rightDF, "left", "ProductID", "ProductID")
	if err != nil {
		fmt.Println("Error in Left Join:", err)
	} else {
		fmt.Println("\nLeft Joined DataFrame:")
		leftJoinedDF.Print()
	}

	// Example 19: Inner joined DF
	innerJoinedDF, err := leftDF.Join(rightDF, "inner", "ProductID", "ProductID")
	if err != nil {
		fmt.Println("Error in Inner Join:", err)
	} else {
		fmt.Println("\nInner Joined DataFrame:")
		innerJoinedDF.Print()
	}

	// Example 20: Outer Join Dataframe
	outerJoinedDF, err := leftDF.Join(rightDF, "outer", "ProductID", "ProductID")
	if err != nil {
		fmt.Println("Error in Outer Join:", err)
	} else {
		fmt.Println("\nOuter Joined DataFrame:")
		outerJoinedDF.Print()
	}

	// Example 21: Reading 100k Rows From CSV And Getting Descriptive Stats
	timeStart := time.Now()
	largeDF, err := ReadCSVToDataFrame("large_datav1.csv")
	if err != nil {
		log.Fatalf("Couldn't Load CSV: %v", err)
	}

	largeDFStats := largeDF.Describe()
	largeDFStats.Print()

	duration := time.Since(timeStart)
	fmt.Printf("\nReading a 100k row csv and getting descriptive stats took %v\n", duration)

	// Example 22: Series Aggregations
	if revenueSeries := largeDF.Col("Revenue"); revenueSeries != nil {
		dfSum, err := revenueSeries.Sum()
		if err != nil {
			fmt.Println("Error calculating sum of Revenue:", err)
		} else {
			fmt.Printf("The sum of Revenue is %.2f\n", dfSum)
		}

		dfAverage, err := revenueSeries.Average()
		if err != nil {
			fmt.Println("Error calculating average of Revenue:", err)
		} else {
			fmt.Printf("The Average of Revenue is %.2f\n", dfAverage)
		}

		dfCount := revenueSeries.CountNonNull()
		fmt.Println("Number of Non-Null values in Revenue:", dfCount)

		lowPercentile, err := revenueSeries.Percentile(1.0)
		if err != nil {
			fmt.Println("Error calculating 1st percentile of Revenue:", err)
		} else {
			fmt.Printf("The 1st percentile of Revenue is %.2f\n", lowPercentile)
		}

		highPercentile, err := revenueSeries.Percentile(99.0)
		if err != nil {
			fmt.Println("Error calculating 99th percentile of Revenue:", err)
		} else {
			fmt.Printf("The 99th percentile of Revenue is %.2f\n", highPercentile)
		}
	} else {
		fmt.Println("Revenue column not found in largeDF")
	}

	// Example 23: Rename a column
	err = df1.Rename("Purchase", "Amount")
	if err != nil {
		fmt.Println("Error renaming column:", err)
	} else {
		fmt.Println("\nDataFrame after renaming 'Purchase' to 'Amount':")
		df1.Head(5).Print()
	}

	// Example 24: Concatenate two DataFrames
	df2 := df1.Head(5) // Create a smaller DataFrame for demonstration
	concatDF := df1.Concat(df2)
	fmt.Println("\nConcatenated DataFrame:")
	concatDF.Tail(10).Print()

	// Example 25: Cumulative sum of a numeric column
	cumsumDF := df1.Cumsum()
	fmt.Println("\nCumulative sum of numeric columns:")
	cumsumDF.Head(10).Print()

	// Example 26: Mode of a column
	if stateSeries := df1.Col("State"); stateSeries != nil {
		modeValues, err := stateSeries.Mode()
		if err != nil {
			fmt.Println("Error calculating mode of State:", err)
		} else {
			fmt.Println("\nMode of 'State' column:", modeValues)
		}
	} else {
		fmt.Println("State column not found")
	}

	if revenueSeries := largeDF.Col("Revenue"); revenueSeries != nil {
		medianRevenue, err := revenueSeries.Median()
		if err != nil {
			fmt.Println("Error calculating median revenue:", err)
		} else {
			fmt.Printf("\nMedian Revenue: %.2f\n", medianRevenue)
		}
	} else {
		fmt.Println("Revenue column not found in largeDF")
	}

	// Example 28: Quantile of Revenue column
	if revenueSeries := largeDF.Col("Revenue"); revenueSeries != nil {
		q75, err := revenueSeries.Quantile(0.75)
		if err != nil {
			fmt.Println("Error calculating 75th percentile of revenue:", err)
		} else {
			fmt.Printf("\n75th percentile of Revenue: %.2f\n", q75)
		}
	} else {
		fmt.Println("Revenue column not found in largeDF")
	}

	// Example 29: Get the n largest values in Revenue column
	if revenueSeries := largeDF.Col("Revenue"); revenueSeries != nil {
		top5Revenue, err := revenueSeries.Nlargest(5)
		if err != nil {
			fmt.Println("Error getting top 5 Revenue values:", err)
		} else {
			fmt.Println("\nTop 5 Revenue values:")
			top5Revenue.Print()
		}
	} else {
		fmt.Println("Revenue column not found in largeDF")
	}

	// Example 30: Get the n smallest values in Revenue column
	if revenueSeries := largeDF.Col("Revenue"); revenueSeries != nil {
		bottom5Revenue, err := revenueSeries.Nsmallest(5)
		if err != nil {
			fmt.Println("Error getting bottom 5 Revenue values:", err)
		} else {
			fmt.Println("\nBottom 5 Revenue values:")
			bottom5Revenue.Print()
		}
	} else {
		fmt.Println("Revenue column not found in largeDF")
	}

	// Example 31: Cumulative sum of Revenue series
	if revenueSeries := largeDF.Col("Revenue"); revenueSeries != nil {
		revenueCumsum := revenueSeries.Cumsum()
		fmt.Println("\nCumulative sum of Revenue:")
		revenueCumsum.Print()
	} else {
		fmt.Println("Revenue column not found in largeDF")
	}
}
