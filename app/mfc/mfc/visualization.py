# mfc/mfc/visualization.py

import dash
import dash_cytoscape as cyto
from dash import html, dcc
import networkx as nx
import logging

class Visualization:
    def __init__(self, pattern_graph: nx.DiGraph):
        self.pattern_graph = pattern_graph
        logging.info("Initialized Visualization class.")

    def visualize_top_n_patterns(self, top_n: int = 20):
        """
        Visualizes the top N patterns based on transition frequency.

        Args:
            top_n (int): Number of top patterns to visualize.
        """
        # Sort edges by weight and select top N
        top_edges = sorted(self.pattern_graph.edges(data=True), key=lambda x: x[2].get('weight', 0), reverse=True)[:top_n]

        # Create a subgraph with top edges
        subgraph = nx.DiGraph()
        subgraph.add_edges_from([(u, v, d) for u, v, d in top_edges])

        # Prepare Cytoscape elements
        cy_elements = []
        for node in subgraph.nodes():
            cy_elements.append({'data': {'id': node, 'label': node}})
        for u, v, d in subgraph.edges(data=True):
            cy_elements.append({'data': {'source': u, 'target': v, 'weight': d.get('weight', 1)}})

        # Initialize Dash app
        app = dash.Dash(__name__)

        app.layout = html.Div([
            html.H1("Top Patterns Visualization"),
            cyto.Cytoscape(
                id='pattern-graph',
                elements=cy_elements,
                layout={'name': 'cose'},
                style={'width': '100%', 'height': '800px'},
                stylesheet=[
                    {
                        'selector': 'node',
                        'style': {
                            'label': 'data(label)',
                            'width': '20px',
                            'height': '20px',
                            'background-color': '#BFD7B5',
                            'font-size': '10px'
                        }
                    },
                    {
                        'selector': 'edge',
                        'style': {
                            'width': '2px',
                            'line-color': '#A3C4BC',
                            'target-arrow-color': '#A3C4BC',
                            'target-arrow-shape': 'triangle',
                            'curve-style': 'bezier'
                        }
                    }
                ]
            ),
            html.Label("Filter by Minimum Weight:"),
            dcc.Slider(
                id='weight-slider',
                min=1,
                max=max([d['weight'] for _, _, d in top_edges]) if top_edges else 10,
                step=1,
                value=1,
                marks={i: str(i) for i in range(1, 11)}
            )
        ])

        @app.callback(
            dash.dependencies.Output('pattern-graph', 'elements'),
            [dash.dependencies.Input('weight-slider', 'value')]
        )
        def update_graph(min_weight):
            filtered_edges = [
                {'data': {'source': u, 'target': v, 'weight': d.get('weight', 1)}}
                for u, v, d in top_edges if d.get('weight', 1) >= min_weight
            ]
            filtered_nodes = list(set([edge['data']['source'] for edge in filtered_edges] +
                                      [edge['data']['target'] for edge in filtered_edges]))
            elements = [{'data': {'id': node, 'label': node}} for node in filtered_nodes] + filtered_edges
            return elements

        logging.info("Launching Dash app for visualization.")
        app.run_server(debug=False)
