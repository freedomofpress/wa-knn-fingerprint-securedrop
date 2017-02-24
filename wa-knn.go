package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sync"
)

func main() {
	fitFlag := flag.Bool("fit", false, `Learn weights based on labeled testing
	data.`)
	predictProbaFlag := flag.Bool("predict-proba", false, `Predict testing data
	class labels using weight slice.`)
	dec := json.NewDecoder(os.Stdin)
	enc := json.NewEncoder(os.Stdout)

	var result []float64
	if *fitFlag {
		result = fit(dec)
	} else if *predictProbaFlag {
		result = predictProba(dec)
	}

	enc.Encode(&result)
}

type FitArgs struct {
	XTrain                        [][]float64
	YTrain                        []float64
	Rounds                        int
	RecoPointsNum                 int // Number of pts that recommend weight changes in each step
	IncreaseWeightsProportionally bool
}

func fit(dec *json.Decoder) (weight []float64) {
	var args FitArgs
	if err := dec.Decode(&args); err != nil {
		fmt.Println(err)
		os.Exit(1)
	}

	weight = wllcc(args.XTrain, args.YTrain, args.Rounds, args.RecoPointsNum, args.IncreaseWeightsProportionally)
	return
}

func getWeightedL1Norm(p1, p2, weight []float64, presentFeats []int) (d float64) {
	for _, v := range presentFeats {
		if p2[v] != 0 {
			d += weight[v] * math.Abs(p1[v]-p2[v])
		}
	}
	return
}

func getMinSubset(distList []float64, subsetInds []int) (idx int, val float64) {
	for _, v := range subsetInds {
		if distList[v] < val {
			idx = v
			val = distList[v]
		}
	}
	return
}

func getMin(distList []float64) (idx int, val float64) {
	for i, v := range distList {
		if v < val {
			idx = i
			val = v
		}
	}
	return
}

func getMax(distList []float64) (idx int, val float64) {
	for i, v := range distList {
		if v > val {
			val = v
			idx = i
		}
	}
	return
}

// Weight Learning by Locally Collapsing Classes
func wllcc(xTrain [][]float64, yTrain []float64, rounds, recoPointsNum int, increaseWeightsProportionally bool) (weight []float64) {
	distList := make([]float64, len(xTrain))
	recoGoodList := make([]int, recoPointsNum)
	recoBadList := make([]int, recoPointsNum)
	featNum := len(xTrain[0])
	featDist := make([]float64, featNum)
	presentFeats := make([]int, featNum)
	badList := make([]float64, featNum)
	var classInds, nonClassInds *[]int

	// We initialize the weights with random values in [0.5,1.5), although Wang
	// noted this has little effect on WLLCC's ability to optimize weights and
	// avoid local minima
	for i := 0; i < featNum; i++ {
		weight[i] = rand.Float64() + 0.5
	}

	// Make slices of the indices of our monitored and non-monitored classes
	var monitoredCount, nonMonitoredCount int
	for _, v := range yTrain {
		if v == 0 {
			monitoredCount++
		} else {
			nonMonitoredCount++
		}
	}
	monitoredInds := make([]int, monitoredCount)
	nonMonitoredInds := make([]int, nonMonitoredCount)
	monitoredCount, nonMonitoredCount = 0, 0
	for i, v := range yTrain {
		if v == 0 {
			monitoredInds[monitoredCount] = i
			monitoredCount++
		} else {
			nonMonitoredInds[nonMonitoredCount] = i
			nonMonitoredCount++
		}
	}

	// Looping over each training point pTrain...
	for i, pTrain := range xTrain {
		// Optimize by finding upfront which features are present for a training
		// point and not checking for the missing ones many times over later.
		var numPresent int
		for j := 0; j < featNum; j++ {
			if pTrain[j] != 0 {
				presentFeats[numPresent] = j
				numPresent++
			}
		}

		if yTrain[i] == 0 {
			classInds = &monitoredInds
			nonClassInds = &nonMonitoredInds
		} else {
			classInds = &nonMonitoredInds
			nonClassInds = &monitoredInds
		}

		// We train on each point rounds times
		for round := 0; round < rounds; round++ {
			// Compute the weighted L^1 norm between pTrain and all other points in
			// xTrain. Parallelized using goroutines.
			var wg sync.WaitGroup
			for j, pPrime := range xTrain {
				wg.Add(1)
				go func(distList, weight, pTrain, pPrime []float64, presentFeats []int, j, numPresent int) {
					defer wg.Done()
					distList[j] = getWeightedL1Norm(pTrain, pPrime, weight, presentFeats[:numPresent])
				}(distList, weight, pTrain, pPrime, presentFeats, j, numPresent)

			}
			// Wait for all norms to finishes computing (goroutines)
			wg.Wait()

			// Don't consider the distance between pTrain and itself
			_, maxVal := getMax(distList)
			distList[i] = maxVal

			// Find the recoPointsNum number of closest instances for the same class
			// as our training point pTrain, recoGoodList, and the closest instances
			// from the other class, recoBadList. TODO: consider replacing this loop
			// construct with a partial heap sort. For smaller datasets this loop is
			// actually more efficient because of the time it takes to create and
			// garbage collect the datastructures, but a partial heap sort could make
			// sense w/ a very large dataset.
			var maxGoodDist float64
			for j := 0; j < recoPointsNum; j++ {
				minIdx, minVal := getMinSubset(distList, *classInds)
				if minVal > maxGoodDist {
					maxGoodDist = minVal
				}
				distList[minIdx] = maxVal // make sure we don't select the same instance again
				recoGoodList[j] = minIdx
			}
			var pointBadness float64
			for j := 0; j < recoPointsNum; j++ {
				minIdx, _ := getMinSubset(distList, *nonClassInds)
				if distList[minIdx] <= maxGoodDist {
					pointBadness++
				}
				distList[minIdx] = maxVal // make sure we don't select the same instance again
				recoBadList[j] = minIdx
			}

			// For each feature...
			for j := 0; j < featNum; j++ {
				// Find maxGoodFeatDist, the maximum distance between pTrain and all
				// points in recoGoodList
				var maxGoodFeatDist float64
				for _, v := range recoGoodList {
					if xTrain[v][j] != 0.0 && pTrain[j] != 0.0 {
						n := weight[j] * math.Abs(xTrain[v][j]-pTrain[j])
						if n > maxGoodFeatDist {
							maxGoodFeatDist = n
						}
					}
				}
				// Find the relevant number of bad distances, badList[j], and the total
				// distance between pTrain[j] and p'[j] for p' in recoBadList,
				// featDist[j]. Side note: because Go garbage collection is slow, it's
				// faster to reset the values in a slice than to let Go garbage collect
				// it initialize a new one with each round.
				badList[j] = 0
				featDist[j] = 0
				for _, v := range recoBadList {
					if xTrain[v][j] != 0.0 && pTrain[j] != 0.0 {
						n := math.Abs(xTrain[v][j] - pTrain[j])
						if weight[j]*n <= maxGoodFeatDist {
							badList[j]++
						}
						featDist[j] += n
					}
				}
			}

			_, minBadList := getMin(badList)
			// Decrease the weights to reduce the distance between pTrain and
			// recoGoodList, while keeping the distance between pTrain and recoBadList
			// the same. c1 can be thought of as the total distance by which we bring
			// pTrain and recoBadList together during the weight reduction step. We
			// reduce the weight of features that are not most useful in
			// distinguishing pTrain from points in recoBadList (i.e., weights j for
			// which badList[j] != minBadList), while increasing equally all weights
			// which are most useful for distinguishing pTrain from points in
			// recoBadList keep the aforementioned overall distance the same.
			var c1 float64
			for j := 0; j < featNum; j++ {
				if badList[j] != minBadList {
					deltaW := weight[j] * 0.01 * badList[j] * (0.2 + pointBadness) / math.Pow(float64(recoPointsNum), 2)
					weight[j] -= deltaW
					c1 += deltaW * featDist[j]
				}
			}
			var totalfd float64
			for j := 0; j < featNum; j++ {
				if badList[j] == minBadList && weight[j] > 0 {
					totalfd += featDist[j]
				}
			}
			if increaseWeightsProportionally {
				// In this variant of the weight increase phase of weight adjustment, we
				// increase useful weights proportionally their current value, while
				// still working within the overall strategy of keeping the total
				// distance between pTrain and recoBadList the same.
				var weightTotal float64
				for j := 0; j < featNum; j++ {
					if badList[j] == minBadList && weight[j] > 0 {
						weightTotal += weight[j]
					}
				}
				weightAverage := weightTotal / float64(featNum)
				for j := 0; j < featNum; j++ {
					if badList[j] == minBadList && weight[j] > 0 {
						weight[j] += (c1 / totalfd) * (weight[j] / weightAverage)
					}
				}
			} else {
				// Original weight increase
				for j := 0; j < featNum; j++ {
					if badList[j] == minBadList && weight[j] > 0 {
						weight[j] += c1 / totalfd
					}
				}
			}
		}
	}
	return
}

type PredictProbaArgs struct {
	XTrain      [][]float64
	YTrain      []float64
	XTest       [][]float64
	Weight      []float64
	NeighborNum int
}

func predictProba(dec *json.Decoder) (resultY []float64) {
	var args PredictProbaArgs
	if err := dec.Decode(&args); err != nil {
		fmt.Println(err)
		os.Exit(1)
	}

	resultY = classify(args.XTrain, args.XTest, args.YTrain, args.Weight, args.NeighborNum)
	return
}

func classify(xTrain, xTest [][]float64, yTrain, weight []float64, neighborNum int) (yResult []float64) {
	distList := make([]float64, len(xTrain))
	featNum := len(xTrain[0])
	presentFeats := make([]int, featNum)

	for i, pTest := range xTest {
		var numPresent int
		for j := 0; j < featNum; j++ {
			if pTest[j] != 0 {
				presentFeats[numPresent] = j
				numPresent++
			}
		}

		var wg sync.WaitGroup
		for j, pPrime := range xTrain {
			wg.Add(1)
			go func(distList, weight, pTest, pPrime []float64, presentFeats []int, j, numPresent int) {
				defer wg.Done()
				distList[j] = getWeightedL1Norm(pTest, pPrime, weight, presentFeats[:numPresent])
			}(distList, weight, pTest, pPrime, presentFeats, j, numPresent)

		}
		// Wait for all norms to finishes computing (goroutines)
		wg.Wait()

		_, maxVal := getMax(distList)

		for j := 0; j < neighborNum; j++ {
			minIdx, _ := getMin(distList)
			if yTrain[minIdx] == 0 {
				yTrain[i] = 0
				break
			}
			distList[minIdx] = maxVal // make sure we don't select the same instance again
		}
		yTrain[i] = 1
	}
	return
}
