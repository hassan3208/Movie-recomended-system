import pandas as pd
import streamlit as st
import functions as f


movies=['enter a movie name']+f.get_movies_list()

st.title('Movie recomended system')

selected_movie = st.selectbox('Choose your movie:', movies,placeholder='enter a movie name')


try:
    recomended=f.recommend(selected_movie)
    
    st.write('we recomend you following movies')
    for i in recomended:
        poster=f.get_poster(i)
        if poster:
            st.image(poster, caption=f'Poster of {i}', use_column_width=True)
        else:
            st.write('Poster not found')
            
except:
    st.write('enter a valid movie name')
        
    




