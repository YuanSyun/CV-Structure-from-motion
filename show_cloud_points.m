function show_cloud_points()

index = 1;

switch index
    case{1}
        file_name = 'Mesona1';
        image_name = './data/Mesona1.JPG';
    case{2}
        file_name = 'Statue1';
        image_name = './data/Statue1.bmp';
    case{3}
        file_name = 'our';
        image_name = './data/our.jpg';
end

three_d_filename = "./data/three_d_points_"+file_name+".csv";
two_d_filename = "./data/two_d_points_"+file_name+".csv";
camera_matrix_filename = "./data/camera_matrix_"+file_name+".csv";

three_d_points = csvread(three_d_filename);
two_d_points = csvread(two_d_filename);
camera_matrix = csvread(camera_matrix_filename);

obj_main(three_d_points, two_d_points, camera_matrix, image_name, 1);