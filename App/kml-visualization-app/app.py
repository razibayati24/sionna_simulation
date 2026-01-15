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
    ui.input_select(
        "year_filter",
        "Filter by Year (or 'All' for all years):",
        choices=["All", "2020", "2021", "2022", "2023", "2024"],
        selected="All"
    ),
    ui.output_ui("map_display"),
    title="5G Coverage Map",
)


def read_geometry_from_table(connection, table_name: str, year_filter: str = "All"):
    """
    Reads geometry data (WKT format) from the Unity Catalog table.
    Returns a pandas DataFrame with year, geometry_wkt, and layer_id columns.
    """
    try:
        with connection.cursor() as cursor:
            # Build query based on year filter
            if year_filter == "All":
                query = f"SELECT year, geometry_wkt, layer_id FROM {table_name} ORDER BY year"
            else:
                query = f"SELECT year, geometry_wkt, layer_id FROM {table_name} WHERE year = {year_filter} ORDER BY year"
            
            print(f"Executing query: {query}")
            cursor.execute(query)
            result = cursor.fetchall_arrow()
            
            if result.num_rows == 0:
                raise ValueError(f"Table {table_name} is empty or no rows found for the selected filter")
            
            # Convert to pandas
            df = result.to_pandas()
            print(f"Retrieved {len(df)} rows from table")
            return df
            
    except Exception as e:
        print(f"Error reading from table: {e}")
        raise e


def visualize_wkt_geometries(df) -> str:
    """
    Parses WKT geometries and creates an HTML map visualization using Folium.
    Returns HTML string with embedded map.
    """
    try:
        import folium
        from shapely import wkt
        import pandas as pd
        
        if df.empty:
            return """
            <div style="padding: 20px; border: 1px solid #ccc; border-radius: 5px;">
                <h3>No data to visualize</h3>
                <p>No geometry data found in the table.</p>
            </div>
            """
        
        # Parse WKT geometries and extract coordinates
        all_coords = []
        polygons_by_year = {}
        
        for idx, row in df.iterrows():
            year = row['year']
            layer_id = row.get('layer_id', f'5g_{year}')
            wkt_str = row['geometry_wkt']
            
            if pd.isna(wkt_str) or not wkt_str:
                continue
            
            try:
                # Parse WKT to shapely geometry
                geom = wkt.loads(str(wkt_str))
                
                # Extract coordinates based on geometry type
                if geom.geom_type == 'MultiPolygon':
                    for polygon in geom.geoms:
                        coords = list(polygon.exterior.coords)
                        # Convert to [lat, lon] format for folium
                        folium_coords = [[lat, lon] for lon, lat in coords]
                        all_coords.extend(folium_coords)
                        
                        if year not in polygons_by_year:
                            polygons_by_year[year] = []
                        polygons_by_year[year].append(folium_coords)
                elif geom.geom_type == 'Polygon':
                    coords = list(geom.exterior.coords)
                    folium_coords = [[lat, lon] for lon, lat in coords]
                    all_coords.extend(folium_coords)
                    
                    if year not in polygons_by_year:
                        polygons_by_year[year] = []
                    polygons_by_year[year].append(folium_coords)
                elif geom.geom_type == 'Point':
                    coords = [geom.x, geom.y]
                    all_coords.append([coords[1], coords[0]])  # [lat, lon]
                    
                    if year not in polygons_by_year:
                        polygons_by_year[year] = []
                    polygons_by_year[year].append([coords[1], coords[0]])
                    
            except Exception as e:
                print(f"Error parsing geometry for year {year}: {e}")
                continue
        
        if not all_coords:
            return """
            <div style="padding: 20px; border: 1px solid #ccc; border-radius: 5px;">
                <h3>No valid geometries found</h3>
                <p>Could not parse geometry data from the table.</p>
            </div>
            """
        
        # Calculate map bounds
        lats = [c[0] for c in all_coords if c[0] is not None]
        lons = [c[1] for c in all_coords if c[1] is not None]
        
        if not lats or not lons:
            center = [37.7749, -122.4194]  # Default center
            zoom = 10
        else:
            center_lat = (min(lats) + max(lats)) / 2
            center_lon = (min(lons) + max(lons)) / 2
            center = [center_lat, center_lon]
            
            # Calculate appropriate zoom level
            lat_range = max(lats) - min(lats)
            lon_range = max(lons) - min(lons)
            max_range = max(lat_range, lon_range)
            
            if max_range > 50:
                zoom = 4
            elif max_range > 10:
                zoom = 5
            elif max_range > 5:
                zoom = 6
            elif max_range > 1:
                zoom = 7
            else:
                zoom = 8
        
        # Create map
        m = folium.Map(location=center, zoom_start=zoom)
        
        # Color scheme for different years
        colors = {
            2020: 'blue',
            2021: 'green',
            2022: 'red',
            2023: 'purple',
            2024: 'orange'
        }
        
        # Add polygons to map, grouped by year
        for year, polygons in polygons_by_year.items():
            color = colors.get(year, 'gray')
            
            for polygon_coords in polygons:
                if len(polygon_coords) > 2:
                    folium.Polygon(
                        locations=polygon_coords,
                        color=color,
                        fill=True,
                        fillColor=color,
                        fillOpacity=0.3,
                        weight=2,
                        popup=f"Year: {year}"
                    ).add_to(m)
        
        # Add legend
        legend_html = '<div style="position: fixed; bottom: 50px; right: 50px; width: 200px; background-color: white; border:2px solid grey; z-index:9999; font-size:14px; padding: 10px;"><h4>Year Legend</h4>'
        for year in sorted(polygons_by_year.keys()):
            color = colors.get(year, 'gray')
            legend_html += f'<p><i class="fa fa-square fa-1x" style="color:{color}"></i> {year}</p>'
        legend_html += '</div>'
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Convert map to HTML
        return m._repr_html_()
        
    except ImportError as ie:
        missing = str(ie).split("'")[1] if "'" in str(ie) else "unknown"
        return f"""
        <div style="padding: 20px; border: 1px solid #ccc; border-radius: 5px;">
            <h3>Missing Required Package</h3>
            <p>Please install '{missing}' package for geometry visualization.</p>
            <p>Import Error: {str(ie)}</p>
        </div>
        """
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return f"""
        <div style="padding: 20px; border: 1px solid #f00; border-radius: 5px; color: #f00;">
            <h3>Error visualizing geometries</h3>
            <p>{str(e)}</p>
            <details>
                <summary>Error Details</summary>
                <pre style="font-size: 12px; max-height: 300px; overflow: auto;">{error_details}</pre>
            </details>
        </div>
        """


# Databricks configuration
cfg = config.Config()

def server(input, output, session):
    # Store geometry data
    geometry_df = reactive.Value(None)
    # Initialize with loading message to ensure something always renders
    map_html = reactive.Value('<div style="padding: 20px;">Loading map...</div>')
    
    @reactive.effect
    @reactive.event(input.year_filter)
    async def load_geometries():
        """Load geometry data from Unity Catalog table asynchronously"""
        try:
            # Get the user access token from the session request header
            user_token = session.http_conn.headers.get('X-Forwarded-Access-Token', None)
            
            # Create a connection with the user's access token
            connection = sql.connect(
                server_hostname=cfg.host,
                http_path=f"/sql/1.0/warehouses/{cfg.warehouse_id}",
                access_token=user_token,
            )
            
            # Get selected year filter
            year_filter = input.year_filter()
            
            # Read geometry data from the table
            df = await asyncio.to_thread(read_geometry_from_table, connection, KML_TABLE, year_filter)
            connection.close()
            
            geometry_df.set(df)
            html = await asyncio.to_thread(visualize_wkt_geometries, df)
            map_html.set(html)
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            error_html = f"""
            <div style="padding: 20px; border: 1px solid #f00; border-radius: 5px; color: #f00;">
                <h3>Error loading geometry data</h3>
                <p><strong>Table:</strong> {KML_TABLE}</p>
                <p><strong>Year Filter:</strong> {input.year_filter()}</p>
                <p><strong>Error:</strong> {str(e)}</p>
                <details>
                    <summary>Error Details</summary>
                    <pre style="font-size: 12px; max-height: 300px; overflow: auto;">{error_details}</pre>
                </details>
                <p><em>Note: Ensure the table exists and is accessible, and contains geometry_wkt column with WKT geometry data.</em></p>
            </div>
            """
            map_html.set(error_html)
            print(f"Error in load_geometries: {e}")
            print(error_details)
    
    @render.ui
    def map_display():
        if map_html() is None:
            return ui.tags.div("Loading map...", style="padding: 20px;")
        return ui.HTML(map_html())


app = App(app_ui, server)

if __name__ == "__main__":
    app.run()

