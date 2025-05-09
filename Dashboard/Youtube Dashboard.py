import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
from plotly.subplots import make_subplots

#%%
pd.set_option('display.max_columns', None)

videos_df = pd.read_csv('/Users/surya/Documents/Class files/Data visualization/FTP/FTP(Surya)/finaldata.csv')


videos_df['publish_date'] = pd.to_datetime(videos_df['publish_date'])
videos_df['year'] = videos_df['publish_date'].dt.year

#%%

# ===============Phase 2 -  Initializing the app ================
app = dash.Dash(__name__)

server = app.server

# ===============Phase 3 -  Defining the layout of the app ================

# Define the layout of the app
app.layout = html.Div([
    html.H1("Youtube Videos Analysis", style={'textAlign': 'center', 'color': 'orange', 'font-size': 30}),
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label = 'Channel Analysis', value='tab-1', style={'backgroundColor': '#AED6F1', 'color': 'black'}, selected_style={'backgroundColor': '#5499C7', 'color': 'white'}),
        dcc.Tab(label='Time Analysis', value='tab-2', style={'backgroundColor': '#AED6F1', 'color': 'black'}, selected_style={'backgroundColor': '#5499C7', 'color': 'white'}),
        dcc.Tab(label='Countries Analysis', value='tab-3', style={'backgroundColor': '#AED6F1', 'color': 'black'}, selected_style={'backgroundColor': '#5499C7', 'color': 'white'})
    ]),
    html.Div(id='layout', style={'backgroundColor': '#F0F8FF'})
])

# Defining layout for tab 1
tab1_layout = html.Div([
            html.H3('Select Channel:'),
            dcc.Dropdown(
                id='channel-dropdown',
                options=[{'label': i, 'value': i} for i in videos_df['channel_title'].unique()],
                value=videos_df['channel_title'].unique()[0],
                multi= True,
                clearable=False
            ),
            html.Div(id='channel-graphs')
        ])

## Defining layout for tab 2

# Define the order for sorting days of the week and times of day
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
time_order = ['Morning', 'Afternoon', 'Evening', 'Night']

# Define layout for tab 2
tab2_layout = html.Div([
    html.H3('Select Day of Week:'),
    dcc.Checklist(
        id='day-of-week-checklist',
        options=[{'label': day, 'value': day} for day in day_order],
        value=['Monday'],  # Default value
        inline=True
    ),
    html.H3('Select Time of Day:'),
    dcc.Checklist(
        id='time-of-day-checklist',
        options=[{'label': time, 'value': time} for time in time_order],
        value=['Morning'],  # Default value
        inline=True
    ),
    html.H3('Select Year Range:'),
    dcc.RangeSlider(
        id='year-range-slider',
        min=videos_df['year'].min(),
        max=videos_df['year'].max(),
        value=[videos_df['year'].min(), videos_df['year'].max()],
        marks={str(year): str(year) for year in sorted(videos_df['year'].unique())},
        step=1
    ),
    html.Div(id='day-time-bar-chart'),
    html.Div(id='year-line-chart')
])

# Define layout for tab 3
tab3_layout = html.Div([
    html.H3('Select Country:'),
    dcc.Checklist(
        id='country-checklist',
        options=[{'label': country, 'value': country} for country in videos_df['publish_country'].unique()],
        value=videos_df['publish_country'].unique()[:3],  # Default to first 3 countries as an example
        inline=True
    ),
    html.Div(id='country-bar-chart'),
    html.Div(id='country-pie-chart'),
    html.Div(id='country-subplots')
])

# ===============Phase 4 -  Defining the callback functions ================

# Define callback for updating tabs
@app.callback(Output('layout', 'children'), Input('tabs', 'value'))
def update_tabs(tab):
    if tab == 'tab-1':
        return tab1_layout
    elif tab == 'tab-2':
        return tab2_layout
    elif tab == 'tab-3':
        return tab3_layout
    else:
        return html.H1('Tab not implemented', style={'color': 'red'})


@app.callback(
    Output('channel-graphs', 'children'),
    [Input('channel-dropdown', 'value')])
def update_graph(selected_channels):
    # Ensure selected_channels is a list even if it is a single value
    if not isinstance(selected_channels, list):
        selected_channels = [selected_channels]

    # Group by 'channel_title' and sum the numeric columns
    channel_totals = videos_df.groupby('channel_title')[['views', 'likes', 'dislikes', 'comment_count']].sum().reset_index()

    # Filter the summarized data for selected channels
    filtered_totals = channel_totals[channel_totals['channel_title'].isin(selected_channels)]

    # Now create a bar chart using Plotly Express for the filtered totals
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Total Views', 'Total Likes', 'Total Dislikes', 'Total Comment Count'),
        vertical_spacing=0.15
    )

    views_fig = px.bar(filtered_totals, x='channel_title', y='views', labels={'x': 'Channel', 'y': 'Total Views'})
    likes_fig = px.bar(filtered_totals, x='channel_title', y='likes', labels={'x': 'Channel', 'y': 'Total Likes'})
    dislikes_fig = px.bar(filtered_totals, x='channel_title', y='dislikes', labels={'x': 'Channel', 'y': 'Total Dislikes'})
    comments_fig = px.bar(filtered_totals, x='channel_title', y='comment_count', labels={'x': 'Channel', 'y': 'Total Comment Count'})

    # Add traces from each figure to the corresponding subplot
    for trace in views_fig.data:
        fig.add_trace(trace, row=1, col=1)
    for trace in likes_fig.data:
        fig.add_trace(trace, row=1, col=2)
    for trace in dislikes_fig.data:
        fig.add_trace(trace, row=2, col=1)
    for trace in comments_fig.data:
        fig.add_trace(trace, row=2, col=2)

    # Update layout
    fig.update_layout(showlegend=False, height=800)

    return dcc.Graph(figure=fig)

# Callback for updating graphs on Tab 2
@app.callback(
    [Output('day-time-bar-chart', 'children'),
     Output('year-line-chart', 'children')],
    [Input('day-of-week-checklist', 'value'),
     Input('time-of-day-checklist', 'value'),
     Input('year-range-slider', 'value')])
def update_day_time_graphs(selected_days, selected_times, selected_years):
    # If no day or time is selected, use the defaults
    selected_days = selected_days or ['Monday']
    selected_times = selected_times or ['Morning']

    # Filter dataframe based on selections
    filtered_df = videos_df[videos_df['published_day_of_week'].isin(selected_days) &
                            videos_df['part_of_day'].isin(selected_times) &
                            videos_df['year'].between(selected_years[0], selected_years[1])]

    # Line chart for total videos by year
    line_fig_data = filtered_df.groupby('year').size().reset_index(name='total_videos')
    line_fig = px.line(line_fig_data, x='year', y='total_videos',
                       labels={'total_videos': 'Total Videos', 'year': 'Year'},
                       title='Total YouTube Videos by Year')

    # Bar chart for total videos by day and time
    bar_fig_data = filtered_df.groupby(['published_day_of_week', 'part_of_day']).size().reset_index(name='total_videos')
    bar_fig_data['published_day_of_week'] = pd.Categorical(bar_fig_data['published_day_of_week'], categories=day_order, ordered=True)
    bar_fig_data['part_of_day'] = pd.Categorical(bar_fig_data['part_of_day'], categories=time_order, ordered=True)
    bar_fig_data = bar_fig_data.sort_values('published_day_of_week')

    bar_fig = px.bar(bar_fig_data, x='published_day_of_week', y='total_videos', color='part_of_day',
                     labels={'total_videos': 'Total Videos', 'published_day_of_week': 'Day of Week'},
                     title='Total YouTube Videos by Day of Week and Time of Day')

    return dcc.Graph(figure=bar_fig), dcc.Graph(figure=line_fig)


# Callback for updating graphs on Tab 3
@app.callback(
    [Output('country-bar-chart', 'children'),
     Output('country-pie-chart', 'children'),
     Output('country-subplots', 'children')],
    [Input('country-checklist', 'value')])
def update_country_graphs(selected_countries):
    # Filter dataframe based on selections
    filtered_df = videos_df[videos_df['publish_country'].isin(selected_countries)]

    # Bar chart for total videos by country with a legend
    bar_fig_data = filtered_df.groupby('publish_country').size().reset_index(name='total_videos')
    bar_fig = px.bar(bar_fig_data, x='publish_country', y='total_videos',
                     title='Total YouTube Videos by Country', color='publish_country')
    # Ensure that legend is displayed for the bar chart
    bar_fig.update_layout(showlegend=True)

    # Pie chart for proportion of total videos by country
    pie_fig = px.pie(bar_fig_data, names='publish_country', values='total_videos',
                     title='Proportion of Total YouTube Videos by Country')

    # Subplots for metrics by day of week
    subplots_fig = make_subplots(rows=2, cols=2,
                                 subplot_titles=['Total Views', 'Total Likes', 'Total Dislikes', 'Total Comment Count'],
                                 horizontal_spacing=0.1)  # Adjust spacing as needed

    # Define colors for countries
    country_colors = px.colors.qualitative.Plotly

    # Add line plots to subplots
    for i, metric in enumerate(['views', 'likes', 'dislikes', 'comment_count']):
        for j, country in enumerate(selected_countries):
            country_metrics = filtered_df[filtered_df['publish_country'] == country]
            country_metrics = country_metrics.groupby(['published_day_of_week', 'publish_country'])[metric].sum().reset_index()
            country_metrics['published_day_of_week'] = pd.Categorical(
                country_metrics['published_day_of_week'], categories=day_order, ordered=True)
            country_metrics = country_metrics.sort_values('published_day_of_week')
            line_fig = px.line(country_metrics, x='published_day_of_week', y=metric, color='publish_country',
                               labels={metric: f'Total {metric.title()}'})
            line_fig.update_traces(line=dict(color=country_colors[j % len(country_colors)]))
            for trace in line_fig.data:
                subplots_fig.add_trace(trace, row=i // 2 + 1, col=i % 2 + 1)

    # Adjust subplots layout for legend positioning
    subplots_fig.update_layout(
        height=800,
        title_text='Metrics by Day of the Week for Selected Countries',
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05
        )
    )

    return dcc.Graph(figure=bar_fig), dcc.Graph(figure=pie_fig), dcc.Graph(figure=subplots_fig)


# ===============Phase 5 -  Running the app ================
if __name__ == '__main__':
    app.run_server(debug=False,
                   port=8000,
                   host='0.0.0.0')