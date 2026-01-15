import asyncio
from shiny import App, ui, render, reactive
from databricks.sdk import config
from databricks import sql
import os

# Defined in `app.yaml`
assert os.getenv("DATABRICKS_WAREHOUSE_ID"), "DATABRICKS_WAREHOUSE_ID must be set in app.yaml."

# Unity Catalog table containing KML data
KML_TABLE = "telecommunications.ca_cell_coverage.ca_5g_coverage"

app_ui = ui.page_fluid(
    ui.h2("5G Coverage Visualization"),
    ui.p("Visualizing coverage data from telecommunications.ca_cell_coverage.ca_5g_coverage"),
    ui.output_ui("map_display"),
    title="5G Coverage Map",
)


def read_kml_from_table(connection, table_name: str) -> str:
    """
    Reads KML content from a Unity Catalog table.
    The table may have KML content in various column formats.
    """
    try:
        with connection.cursor() as cursor:
            # First, try to get the table schema to understand the structure
            cursor.execute(f"DESCRIBE TABLE {table_name}")
            schema = cursor.fetchall_arrow()
            
            # Get column names
            column_names = schema['col_name'].to_pylist()
            print(f"Table columns: {column_names}")
            
            # Try to find a column that likely contains KML content
            # Common column names: kml_content, content, kml, data, body, text
            kml_column = None
            for col in ['kml_content', 'content', 'kml', 'data', 'body', 'text', 'xml']:
                if col in column_names:
                    kml_column = col
                    break
            
            if kml_column:
                # If we found a likely column, query it
                query = f"SELECT {kml_column} FROM {table_name} LIMIT 1"
            else:
                # If no obvious column, try to get all columns and concatenate string columns
                # or just get the first row
                query = f"SELECT * FROM {table_name} LIMIT 1"
            
            print(f"Executing query: {query}")
            cursor.execute(query)
            result = cursor.fetchall_arrow()
            
            if result.num_rows == 0:
                raise ValueError(f"Table {table_name} is empty or no rows found")
            
            # Convert to pandas for easier manipulation
            df = result.to_pandas()
            
            # If we found a specific KML column, use it
            if kml_column:
                kml_content = df[kml_column].iloc[0]
                if kml_content:
                    return str(kml_content)
            
            # Otherwise, try to find any string column that looks like XML/KML
            for col in df.columns:
                value = df[col].iloc[0]
                if isinstance(value, str) and ('<?xml' in value or '<kml' in value.lower() or '<Document' in value):
                    return str(value)
            
            # If still not found, concatenate all string columns
            kml_parts = []
            for col in df.columns:
                value = df[col].iloc[0]
                if isinstance(value, str) and len(value) > 100:  # Likely contains substantial data
                    kml_parts.append(str(value))
            
            if kml_parts:
                return '\n'.join(kml_parts)
            
            # Last resort: return the entire row as JSON/string representation
            raise ValueError(
                f"Could not find KML content in table {table_name}. "
                f"Available columns: {column_names}. "
                f"Please ensure the table contains a column with KML/XML data."
            )
            
    except Exception as e:
        print(f"Error reading from table: {e}")
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


# Databricks configuration
cfg = config.Config()

def server(input, output, session):
    # Store KML content in a reactive value
    kml_content = reactive.Value(None)
    # Initialize with loading message to ensure something always renders
    map_html = reactive.Value('<div style="padding: 20px;">Loading map...</div>')
    
    @reactive.effect
    async def load_kml():
        """Load KML data from Unity Catalog table asynchronously"""
        try:
            # Get the user access token from the session request header
            user_token = session.http_conn.headers.get('X-Forwarded-Access-Token', None)
            
            # Create a connection with the user's access token
            connection = sql.connect(
                server_hostname=cfg.host,
                http_path=f"/sql/1.0/warehouses/{cfg.warehouse_id}",
                access_token=user_token,
            )
            
            # Read KML content from the table
            content = await asyncio.to_thread(read_kml_from_table, connection, KML_TABLE)
            connection.close()
            
            kml_content.set(content)
            html = await asyncio.to_thread(parse_and_visualize_kml, content)
            map_html.set(html)
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            error_html = f"""
            <div style="padding: 20px; border: 1px solid #f00; border-radius: 5px; color: #f00;">
                <h3>Error loading KML data</h3>
                <p><strong>Table:</strong> {KML_TABLE}</p>
                <p><strong>Error:</strong> {str(e)}</p>
                <details>
                    <summary>Error Details</summary>
                    <pre style="font-size: 12px; max-height: 300px; overflow: auto;">{error_details}</pre>
                </details>
                <p><em>Note: Ensure the table exists and is accessible, and contains a column with KML/XML data.</em></p>
            </div>
            """
            map_html.set(error_html)
            print(f"Error in load_kml: {e}")
            print(error_details)
    
    @render.ui
    def map_display():
        if map_html() is None:
            return ui.tags.div("Loading map...", style="padding: 20px;")
        return ui.HTML(map_html())


app = App(app_ui, server)

if __name__ == "__main__":
    app.run()

