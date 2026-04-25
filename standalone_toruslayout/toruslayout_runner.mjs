import { PseudoRandom } from "./descent.ts";
import { TorusLayout } from "./toruslayout.ts";

function wrap01(value) {
  const wrapped = value % 1;
  return wrapped < 0 ? wrapped + 1 : wrapped;
}

function wrapToFundamentalDomain(value, minValue, period) {
  return minValue + wrap01((value - minValue) / period) * period;
}

function buildAdjacency(nodeCount, links) {
  const adjacency = Array.from({ length: nodeCount }, () => []);
  for (const link of links) {
    const source = typeof link.source === "number" ? link.source : link.source.index;
    const target = typeof link.target === "number" ? link.target : link.target.index;
    const weight = Number.isFinite(link.weight) ? Number(link.weight) : 1;
    adjacency[source].push({ target, weight });
    adjacency[target].push({ target: source, weight });
  }
  return adjacency;
}

function allWeightsAreOne(adjacency) {
  return adjacency.every((neighbors) =>
    neighbors.every((edge) => edge.weight === 1)
  );
}

function bfsDistances(adjacency, source) {
  const distances = new Array(adjacency.length).fill(Infinity);
  const queue = [source];
  let queueIndex = 0;
  distances[source] = 0;
  while (queueIndex < queue.length) {
    const node = queue[queueIndex++];
    const baseDistance = distances[node];
    for (const edge of adjacency[node]) {
      if (distances[edge.target] !== Infinity) {
        continue;
      }
      distances[edge.target] = baseDistance + 1;
      queue.push(edge.target);
    }
  }
  return distances;
}

function dijkstraDistances(adjacency, source) {
  const visited = new Array(adjacency.length).fill(false);
  const distances = new Array(adjacency.length).fill(Infinity);
  distances[source] = 0;

  for (let iteration = 0; iteration < adjacency.length; iteration += 1) {
    let bestNode = -1;
    let bestDistance = Infinity;
    for (let node = 0; node < adjacency.length; node += 1) {
      if (!visited[node] && distances[node] < bestDistance) {
        bestDistance = distances[node];
        bestNode = node;
      }
    }
    if (bestNode < 0) {
      break;
    }
    visited[bestNode] = true;
    for (const edge of adjacency[bestNode]) {
      const candidate = bestDistance + edge.weight;
      if (candidate < distances[edge.target]) {
        distances[edge.target] = candidate;
      }
    }
  }

  return distances;
}

function computeShortestPathLengths(graph) {
  if (graph.shortestPathLengths) {
    return graph.shortestPathLengths;
  }

  const adjacency = buildAdjacency(graph.nodes.length, graph.links);
  const algorithm = allWeightsAreOne(adjacency) ? bfsDistances : dijkstraDistances;
  return adjacency.map((_, source) => algorithm(adjacency, source));
}

function prepareGraph(graph, configuration) {
  const width = configuration.svgWidth;
  const height = configuration.svgHeight;
  graph.nodes.forEach((node, index) => {
    node.index = index;
    if (typeof node.x === "undefined") {
      node.x = width / 2;
      node.y = height / 2;
    }
  });
  graph.links.forEach((link) => {
    if (typeof link.source === "number") {
      link.source = graph.nodes[link.source];
    }
    if (typeof link.target === "number") {
      link.target = graph.nodes[link.target];
    }
  });
}

function seedPositions(layout, initialPositions) {
  const graph = layout.graph;
  const configuration = layout.configuration;
  const periodX = configuration.svgWidth / 3;
  const periodY = configuration.svgHeight / 3;
  const minX = periodX;
  const minY = periodY;

  if (initialPositions) {
    if (initialPositions.length !== graph.nodes.length) {
      throw new Error(
        `Expected ${graph.nodes.length} initial positions, got ${initialPositions.length}`
      );
    }
    initialPositions.forEach(([x, y], index) => {
      graph.nodes[index].x = minX + wrap01(Number(x)) * periodX;
      graph.nodes[index].y = minY + wrap01(Number(y)) * periodY;
    });
    return;
  }

  const width = configuration.svgWidth;
  const height = configuration.svgHeight;
  for (let index = 0; index < graph.nodes.length; index += 1) {
    graph.nodes[index].x = width / 2 - 0.5 + layout.random.getNextBetween(0, 1);
    graph.nodes[index].y = height / 2 - 0.5 + layout.random.getNextBetween(0, 1);
  }
}

function startLayout(layout, initialPositions) {
  prepareGraph(layout.graph, layout.configuration);
  seedPositions(layout, initialPositions);
  layout.run();
}

function normalizePositions(graph, configuration) {
  const periodX = configuration.svgWidth / 3;
  const periodY = configuration.svgHeight / 3;
  const minX = periodX;
  const minY = periodY;
  return graph.nodes.map((node) => [
    wrap01((node.x - minX) / periodX),
    wrap01((node.y - minY) / periodY),
  ]);
}

function collectRawPositions(graph, configuration) {
  const periodX = configuration.svgWidth / 3;
  const periodY = configuration.svgHeight / 3;
  const minX = periodX;
  const minY = periodY;
  return graph.nodes.map((node) => [
    wrapToFundamentalDomain(node.x, minX, periodX),
    wrapToFundamentalDomain(node.y, minY, periodY),
  ]);
}

async function readStdin() {
  const chunks = [];
  for await (const chunk of process.stdin) {
    chunks.push(chunk);
  }
  return Buffer.concat(chunks).toString("utf8");
}

const payload = JSON.parse(await readStdin());
payload.graph.shortestPathLengths = computeShortestPathLengths(payload.graph);

const layout = new TorusLayout(payload.graph, payload.config, () => {});
layout.random = new PseudoRandom(payload.seed ?? 1);
startLayout(layout, payload.initialPositions ?? null);

process.stdout.write(
  JSON.stringify({
    positions: normalizePositions(payload.graph, payload.config),
    rawPositions: collectRawPositions(payload.graph, payload.config),
    iterations: layout.step,
    config: payload.config,
  })
);
