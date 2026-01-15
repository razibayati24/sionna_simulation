import asyncio
from shiny import App, ui, render, reactive
import os

# KML file path from Databricks Unity Catalog Volume
KML_FILE_PATH = "/Volumes/telecommunications/ca_cell_coverage/5g_coverage/5G2024.kml"

app_ui = ui.page_fluid(
    ui.h2("5G Coverage Visualization"),
    ui.p("Visualizing coverage data from 5G2024.kml"),
    ui.output_ui("map_display"),
    title="5G Coverage Map",
)


def read_kml_file(file_path: str) -> str:
    """
    Reads a KML file from Databricks Volume.
    In Databricks, Volumes are accessible via standard file system paths.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading KML file: {e}")
        # If direct file access doesn't work, we might need to use dbutils
        # which is only available in notebooks, so we'll handle that case
        raise e


def parse_and_visualize_kml(kml_content: str) -> str:
    """
    Parses KML content and creates an HTML map visualization.
    Returns HTML string with embedded map.
    """
    try:
        # Try to use fastkml and folium if available
        try:
            from fastkml import kml
            import folium
            
            # Parse KML using fastkml
            k = kml.KML()
            k.from_string(kml_content.encode('utf-8'))
            
            # Extract features and coordinates
            features = list(k.features())
            coordinates_list = []
            bounds = []
            
            def extract_coordinates(element):
                """Recursively extract coordinates from KML features"""
                if hasattr(element, 'geometry') and element.geometry:
                    geom = element.geometry
                    # Handle different geometry types
                    if hasattr(geom, 'exterior'):
                        # Polygon
                        if hasattr(geom.exterior, 'coords'):
                            coords = list(geom.exterior.coords)
                            coordinates_list.append(('polygon', coords))
                            for coord in coords:
                                if len(coord) >= 2:
                                    bounds.append([coord[1], coord[0]])  # lat, lon
                    elif hasattr(geom, 'coords'):
                        # LineString or Point
                        coords = list(geom.coords)
                        if len(coords) > 1:
                            coordinates_list.append(('linestring', coords))
                        else:
                            coordinates_list.append(('point', coords))
                        for coord in coords:
                            if len(coord) >= 2:
                                bounds.append([coord[1], coord[0]])  # lat, lon
                
                # Recursively process features
                if hasattr(element, 'features'):
                    for feature in element.features():
                        extract_coordinates(feature)
            
            for feature in features:
                extract_coordinates(feature)
            
            # Create map
            if bounds:
                # Calculate center and bounds
                lats = [b[0] for b in bounds if b[0] is not None]
                lons = [b[1] for b in bounds if b[1] is not None]
                if lats and lons:
                    center_lat = (min(lats) + max(lats)) / 2
                    center_lon = (min(lons) + max(lons)) / 2
                    m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
                    
                    # Add features to map
                    for geom_type, coord_list in coordinates_list:
                        if geom_type == 'polygon':
                            folium.Polygon(
                                locations=[[c[1], c[0]] for c in coord_list if len(c) >= 2],
                                color='blue',
                                fill=True,
                                fillColor='blue',
                                fillOpacity=0.3,
                                popup="Coverage Area"
                            ).add_to(m)
                        elif geom_type == 'linestring':
                            folium.PolyLine(
                                locations=[[c[1], c[0]] for c in coord_list if len(c) >= 2],
                                color='blue',
                                popup="Coverage Line"
                            ).add_to(m)
                        elif geom_type == 'point' and coord_list:
                            coord = coord_list[0]
                            if len(coord) >= 2:
                                folium.Marker(
                                    location=[coord[1], coord[0]],
                                    popup="Coverage Point"
                                ).add_to(m)
                else:
                    # Fallback: create a default map
                    m = folium.Map(location=[37.7749, -122.4194], zoom_start=10)
            else:
                # Fallback: create a default map
                m = folium.Map(location=[37.7749, -122.4194], zoom_start=10)
            
            # Convert map to HTML
            return m._repr_html_()
            
        except ImportError as ie:
            # Fallback: basic HTML display if libraries aren't available
            return f"""
            <div style="padding: 20px; border: 1px solid #ccc; border-radius: 5px;">
                <h3>KML File Content</h3>
                <pre style="max-height: 600px; overflow: auto;">{kml_content[:5000]}</pre>
                <p><em>Note: Install 'fastkml' and 'folium' packages for full map visualization</em></p>
                <p>Import Error: {str(ie)}</p>
            </div>
            """
    except Exception as e:
        return f"""
        <div style="padding: 20px; border: 1px solid #f00; border-radius: 5px; color: #f00;">
            <h3>Error visualizing KML</h3>
            <p>{str(e)}</p>
            <pre style="font-size: 12px;">{str(type(e).__name__)}</pre>
        </div>
        """


def server(input, output, session):
    # Store KML content in a reactive value
    kml_content = reactive.Value(None)
    # Initialize with loading message to ensure something always renders
    map_html = reactive.Value('<div style="padding: 20px;">Loading map...</div>')
    
    @reactive.effect
    async def load_kml():
        """Load KML file asynchronously"""
        try:
            content = await asyncio.to_thread(read_kml_file, KML_FILE_PATH)
            kml_content.set(content)
            html = await asyncio.to_thread(parse_and_visualize_kml, content)
            map_html.set(html)
        except Exception as e:
            error_html = f"""
            <div style="padding: 20px; border: 1px solid #f00; border-radius: 5px; color: #f00;">
                <h3>Error loading KML file</h3>
                <p>Path: {KML_FILE_PATH}</p>
                <p>Error: {str(e)}</p>
                <p><em>Note: Ensure the file path is correct and accessible from the Databricks environment.</em></p>
            </div>
            """
            map_html.set(error_html)
    
    @render.ui
    def map_display():
        if map_html() is None:
            return ui.tags.div("Loading map...", style="padding: 20px;")
        return ui.HTML(map_html())


app = App(app_ui, server)

if __name__ == "__main__":
    app.run()

