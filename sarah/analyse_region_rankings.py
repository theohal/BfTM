import data_loader
from numpy import mean


def getRegionStatistics(optimisation_results):
    region_statistics = {}
    for result in optimisation_results:
        for region in result['used_regions']:
            if region not in region_statistics.keys():
                region_statistics[region] = { 'times_used' : 1, 'test_scores' : [result['test_score']], 'iterations' : [result['iteration']] }
                continue

            region_statistics[region]['times_used'] = region_statistics[region]['times_used'] + 1
            region_statistics[region]['test_scores'].append(result['test_score'])
            region_statistics[region]['iterations'].append(result['iteration'])

    print("All used regions and their rankings:")
    print(region_statistics)
    print("---------------------------------------------------")

    return region_statistics



def sortRegionsByTimesUsed(region_statistics):
    return sorted(region_statistics.items(), key=lambda item: item[1]['times_used'], reverse=True)

def sortRegionsByHighestScore(region_statistics):
    return sorted(region_statistics.items(), key=lambda item: mean(item[1]['test_scores']), reverse=True)


def printTopRankedRegions(region_rankings, top=None):
    if (top == None): top = len(region_rankings)
    print("Top {} (out of {}) used regions:".format(top, len(region_rankings)))

    for rank, region_ranking in enumerate(region_rankings[:top], start=1):
        print("#{}: {} (used {} times, mean_test_score: {})".format(
            rank,
            region_ranking[0],
            region_ranking[1]['times_used'],
            mean(region_ranking[1]['test_scores'])))

    print("---------------------------------------------------")



def getIterationStatisticsForTopRankedRegions(region_rankings, top=None):
    if (top == None): top = len(region_rankings)

    iteration_statistics = {}

    for index, (region, region_ranking) in enumerate(region_rankings):
        if index == top: break # only take top regions into account

        for iteration in region_ranking['iterations']:
            if iteration not in iteration_statistics.keys():
                iteration_statistics[iteration] = [region]
                continue

            iteration_statistics[iteration].append(region)

    return iteration_statistics
 
def sortIterationsByNumberOfTopRankedRegionsUsed(iteration_statistics, top=None):
    return sorted(iteration_statistics.items(), key=lambda item: len(item[1]), reverse=True)

def printTopRankedIterations(iteration_rankings, top=None):
    if top == None: top = len(iteration_rankings)
    print("Top {} (out of {}) iterations:".format(top, len(iteration_rankings)))

    for rank, (iteration, regions) in enumerate(iteration_rankings[:top], start=1):
        print("#{}: iteration {} used {} of the top regions".format(
            rank,
            iteration,
            len(regions)))

    print("---------------------------------------------------")


def extractRegionRankings():
    optimisation_results = data_loader.load_optimisation_results()

    region_statistics = getRegionStatistics(optimisation_results)
    return region_statistics
 

if __name__ == "__main__":
    # find the regions that have been used most across all iterations ("top regions") 
    # to re-use previously stored region_statistics, change to: region_statistics = data_loader.load_region_rankings()
    region_statistics = extractRegionRankings()

    region_rankings = sortRegionsByTimesUsed(region_statistics)
    data_loader.save_region_rankings(region_rankings)

    printTopRankedRegions(region_rankings, top=10)

    # find the iterations that used most of the "top regions" (most used regions across all iterations)
    iteration_statistics = getIterationStatisticsForTopRankedRegions(region_rankings, top=100)
    
    iteration_rankings = sortIterationsByNumberOfTopRankedRegionsUsed(iteration_statistics)
    data_loader.save_iteration_rankings(iteration_rankings)
    
    printTopRankedIterations(iteration_rankings, top=10)
